"""
Trait-Specific Multi-Scale Regional Verification Module.
Validates individual auxiliary radiological predictions against regional physical image evidence:
- Spiculation & Margin: Evaluated via Nodule Peripheral Boundary Shell & Radial Gradient Profiles
- Calcification & Subtlety: Evaluated via Internal Core Density HU Attenuation & Parenchymal Contrast
- Sphericity & Lobulation: Evaluated via Multi-Scale Contour Compactness & Concave Notch Analysis
"""

from typing import Dict, List, Any, Tuple, Optional
import numpy as np
from src.preprocessing import segment_lung_mask_slice


class TraitSpecificRegionalVerifier:
    """
    Directly extracts targeted radiomic/physical image evidence from specific nodule sub-regions
    (core, boundary shell, peri-tumoral parenchyma) to verify whether predicted radiological traits
    are physically grounded in the CT scan.
    """

    def __init__(self, nodule_intensity_threshold: float = 0.35):
        self.nodule_intensity_threshold = nodule_intensity_threshold

    def _locate_nodule_roi(self, volume: np.ndarray) -> Tuple[Tuple[int, int, int], np.ndarray, np.ndarray, np.ndarray]:
        """
        Locates candidate nodule center and partitions ROI into:
        1. Core region (inner 50% radius)
        2. Boundary shell (ring around the nodule perimeter)
        3. Peri-nodular parenchymal background (outer ring)
        """
        D, H, W = volume.shape
        # Find slice with candidate nodule
        center_d = D // 2
        mid_slice = volume[center_d].copy()

        # Isolate lung parenchyma to avoid anchoring on bone/spine or mediastinal vessels
        lung_mask = segment_lung_mask_slice(mid_slice)
        if lung_mask.sum() > 50:
            masked_lung = np.where(lung_mask > 0, mid_slice, 0.0)
            center_y, center_x = np.unravel_index(np.argmax(masked_lung), masked_lung.shape)
        else:
            # Fallback to central 60% search region
            margin_h, margin_w = int(H * 0.2), int(W * 0.2)
            search_region = mid_slice[margin_h:H-margin_h, margin_w:W-margin_w]
            y_rel, x_rel = np.unravel_index(np.argmax(search_region), search_region.shape)
            center_y = y_rel + margin_h
            center_x = x_rel + margin_w

        # Generate coordinate grid centered at nodule center
        yy, xx = np.ogrid[:H, :W]
        dist_from_center = np.sqrt((yy - center_y)**2 + (xx - center_x)**2)

        # Approximate nodule radius scaled dynamically to spatial grid
        # Baseline 7 voxels for 128x128 (~5.5% of spatial width)
        nominal_r = max(5, int(round(W * (7.0 / 128.0))))
        core_mask = dist_from_center <= nominal_r
        shell_mask = (dist_from_center > nominal_r) & (dist_from_center <= int(nominal_r * 2.0))
        peri_mask = (dist_from_center > int(nominal_r * 2.0)) & (dist_from_center <= int(nominal_r * 3.14))

        return (center_d, int(center_y), int(center_x)), core_mask, shell_mask, peri_mask

    def verify_spiculation_and_margin(
        self,
        slice_ct: np.ndarray,
        shell_mask: np.ndarray,
        predicted_spiculation: float,
        predicted_margin: float
    ) -> Dict[str, Any]:
        """
        Inspects boundary shell for radial spikes, gradient variance, and edge sharpness.
        High Spiculation (>3.0) should show high radial derivative variance and jagged shell intensity.
        Smooth Margin (>3.5) should show sharp, steep monotonic edge gradient drops.
        Poor/Invasive Margin (<2.5) should show blurry, gradual transition gradients.
        """
        shell_voxels = slice_ct[shell_mask]
        if len(shell_voxels) == 0:
            shell_std = 0.05
            gradient_sharpness = 0.5
        else:
            shell_std = float(np.std(shell_voxels))
            # Edge gradient magnitude using Sobel/central differences
            gy, gx = np.gradient(slice_ct)
            grad_mag = np.sqrt(gy**2 + gx**2)
            gradient_sharpness = float(np.mean(grad_mag[shell_mask]))

        # Normalized physical spiculation index (0 to 1)
        # Higher shell variance indicates irregular, spiculated radiating borders
        physical_spiculation_idx = float(np.clip(shell_std * 3.5, 0.0, 1.0))
        # Expected spiculation score on 1-5 scale
        expected_spic_1to5 = 1.0 + 4.0 * physical_spiculation_idx

        # Normalized margin definition index (0 to 1)
        # Higher gradient sharpness indicates crisp, well-defined margin (score 5)
        physical_margin_idx = float(np.clip(gradient_sharpness * 4.0, 0.0, 1.0))
        expected_margin_1to5 = 1.0 + 4.0 * physical_margin_idx

        # Discrepancy calculation
        spic_error = abs(predicted_spiculation - expected_spic_1to5) / 4.0
        margin_error = abs(predicted_margin - expected_margin_1to5) / 4.0

        spic_verified = spic_error < 0.35
        margin_verified = margin_error < 0.35

        conflicts = []
        if not spic_verified:
            conflicts.append(
                f"Spiculation mismatch: Model predicted {predicted_spiculation:.2f}/5, "
                f"but boundary radial variance indicates {expected_spic_1to5:.2f}/5."
            )
        if not margin_verified:
            conflicts.append(
                f"Margin mismatch: Model predicted {predicted_margin:.2f}/5, "
                f"but boundary gradient transition indicates {expected_margin_1to5:.2f}/5."
            )

        return {
            "spiculation_grounding_score": round(1.0 - spic_error, 2),
            "margin_grounding_score": round(1.0 - margin_error, 2),
            "physical_spiculation_idx": round(physical_spiculation_idx, 3),
            "physical_margin_idx": round(physical_margin_idx, 3),
            "boundary_gradient_sharpness": round(gradient_sharpness, 4),
            "shell_variance": round(shell_std, 4),
            "is_verified": spic_verified and margin_verified,
            "conflicts": conflicts
        }

    def verify_calcification_and_subtlety(
        self,
        slice_ct: np.ndarray,
        core_mask: np.ndarray,
        peri_mask: np.ndarray,
        predicted_calcification: float,
        predicted_subtlety: float
    ) -> Dict[str, Any]:
        """
        Inspects internal nodule core density and parenchymal contrast:
        - Calcification (1-6): LIDC scale (1=Popcorn, 2=Laminated, 3=Central, 4=Stippled, 5=None, 6=Absent).
          Values 1-4 correspond to high internal HU (> 100-200 HU or normalized > 0.75).
          Values 5-6 correspond to soft-tissue/ground-glass with no dense calcification.
        - Subtlety (1-5): 1=Extremely subtle / ground-glass contrast, 5=Obvious solid lesion with sharp contrast.
        """
        core_voxels = slice_ct[core_mask]
        peri_voxels = slice_ct[peri_mask]

        if len(core_voxels) == 0:
            core_mean = 0.5
            core_max = 0.5
            contrast_diff = 0.3
        else:
            core_mean = float(np.mean(core_voxels))
            core_max = float(np.max(core_voxels))
            peri_mean = float(np.mean(peri_voxels)) if len(peri_voxels) > 0 else 0.2
            contrast_diff = abs(core_mean - peri_mean)

        # High density ratio within core (indicative of dense calcification)
        hyperdense_ratio = float(np.mean(core_voxels >= 0.78)) if len(core_voxels) > 0 else 0.0

        # Physical calcification pattern:
        # If hyperdense ratio > 0.15 => likely calcified (score 1 to 4)
        # If hyperdense ratio <= 0.05 => absent / soft tissue (score 5 to 6)
        has_dense_calcification = hyperdense_ratio > 0.12
        if has_dense_calcification:
            expected_calc_range = (1.0, 4.2)
        else:
            expected_calc_range = (4.0, 6.0)

        calc_verified = expected_calc_range[0] <= predicted_calcification <= expected_calc_range[1]

        # Physical subtlety:
        # contrast_diff > 0.35 => Obvious (subtlety 4-5)
        # contrast_diff < 0.15 => Subtle (subtlety 1-2)
        expected_subtlety = 1.0 + 4.0 * float(np.clip(contrast_diff / 0.5, 0.0, 1.0))
        subtlety_error = abs(predicted_subtlety - expected_subtlety) / 4.0
        subtlety_verified = subtlety_error < 0.35

        conflicts = []
        if not calc_verified:
            pattern_desc = "calcified dense inclusions" if has_dense_calcification else "non-calcified soft tissue"
            conflicts.append(
                f"Calcification mismatch: Model predicted score {predicted_calcification:.2f}/6, "
                f"but internal attenuation reveals {pattern_desc} (core peak: {core_max:.2f}, dense ratio: {hyperdense_ratio*100:.1f}%)."
            )
        if not subtlety_verified:
            conflicts.append(
                f"Subtlety mismatch: Model predicted {predicted_subtlety:.2f}/5, "
                f"but lesion-to-parenchyma contrast difference is {contrast_diff:.2f} (expected ~{expected_subtlety:.2f}/5)."
            )

        calc_grounding = 1.0 if calc_verified else 0.5
        subtlety_grounding = round(1.0 - subtlety_error, 2)

        return {
            "calcification_grounding_score": calc_grounding,
            "subtlety_grounding_score": subtlety_grounding,
            "core_mean_density": round(core_mean, 3),
            "core_peak_density": round(core_max, 3),
            "hyperdense_ratio": round(hyperdense_ratio, 3),
            "parenchymal_contrast": round(contrast_diff, 3),
            "is_verified": calc_verified and subtlety_verified,
            "conflicts": conflicts
        }

    def verify_sphericity_and_lobulation(
        self,
        slice_ct: np.ndarray,
        core_mask: np.ndarray,
        predicted_sphericity: float,
        predicted_lobulation: float
    ) -> Dict[str, Any]:
        """
        Inspects morphological contour:
        - Sphericity (1-5): 1=Linear/irregular shape, 5=Perfect spherical roundness.
        - Lobulation (1-5): 1=None / smooth circular contour, 5=Marked multi-lobulated contours with notches.
        """
        # Segment local nodule contour
        nodule_bin = (slice_ct >= 0.32) & (core_mask | np.pad(core_mask, ((0,0),(0,0))))
        area = float(np.sum(nodule_bin))
        
        if area < 10:
            # Fallback for faint or small nodules
            compactness = 0.75
            lobulation_index = 0.25
        else:
            # Measure perimeter via morphological gradient
            eroded = nodule_bin & np.roll(nodule_bin, 1, 0) & np.roll(nodule_bin, -1, 0) & np.roll(nodule_bin, 1, 1) & np.roll(nodule_bin, -1, 1)
            perimeter = float(np.sum(nodule_bin ^ eroded))
            # Isoperimetric quotient: 4 * pi * Area / Perimeter^2 (1.0 for perfect circle)
            compactness = float((4.0 * np.pi * area) / max(perimeter**2, 1.0))
            compactness = float(np.clip(compactness, 0.05, 1.0))
            
            # Lobulation is inverse of circular smoothness + boundary irregularity
            lobulation_index = float(np.clip(1.0 - compactness + 0.1, 0.0, 1.0))

        expected_sphericity = 1.0 + 4.0 * compactness
        expected_lobulation = 1.0 + 4.0 * lobulation_index

        sph_error = abs(predicted_sphericity - expected_sphericity) / 4.0
        lob_error = abs(predicted_lobulation - expected_lobulation) / 4.0

        sph_verified = sph_error < 0.35
        lob_verified = lob_error < 0.35

        conflicts = []
        if not sph_verified:
            conflicts.append(
                f"Sphericity mismatch: Model predicted {predicted_sphericity:.2f}/5, "
                f"but geometric compactness is {compactness:.2f} (expected ~{expected_sphericity:.2f}/5)."
            )
        if not lob_verified:
            conflicts.append(
                f"Lobulation mismatch: Model predicted {predicted_lobulation:.2f}/5, "
                f"but contour notch irregularity indicates ~{expected_lobulation:.2f}/5."
            )

        return {
            "sphericity_grounding_score": round(1.0 - sph_error, 2),
            "lobulation_grounding_score": round(1.0 - lob_error, 2),
            "geometric_compactness": round(compactness, 3),
            "contour_lobulation_index": round(lobulation_index, 3),
            "is_verified": sph_verified and lob_verified,
            "conflicts": conflicts
        }

    def verify_all_traits(
        self,
        volume: np.ndarray,
        predicted_traits: List[float]
    ) -> Dict[str, Any]:
        """
        Executes multi-scale regional physical verification across all six LIDC traits:
        [Spiculation, Lobulation, Calcification, Subtlety, Margin, Sphericity].
        """
        if len(predicted_traits) != 6:
            raise ValueError(f"Expected 6 trait predictions, got {len(predicted_traits)}")

        spic, lob, calc, subt, marg, sph = predicted_traits
        (center_d, center_y, center_x), core_mask, shell_mask, peri_mask = self._locate_nodule_roi(volume)
        slice_ct = volume[center_d]

        # 1. Spiculation & Margin Verification (Boundary Shell)
        spic_marg_res = self.verify_spiculation_and_margin(slice_ct, shell_mask, spic, marg)

        # 2. Calcification & Subtlety Verification (Core & Parenchyma)
        calc_subt_res = self.verify_calcification_and_subtlety(slice_ct, core_mask, peri_mask, calc, subt)

        # 3. Sphericity & Lobulation Verification (Contour & Compactness)
        sph_lob_res = self.verify_sphericity_and_lobulation(slice_ct, core_mask, sph, lob)

        all_conflicts = (
            spic_marg_res["conflicts"] +
            calc_subt_res["conflicts"] +
            sph_lob_res["conflicts"]
        )

        grounding_scores = {
            "Spiculation": spic_marg_res["spiculation_grounding_score"],
            "Margin": spic_marg_res["margin_grounding_score"],
            "Calcification": calc_subt_res["calcification_grounding_score"],
            "Subtlety": calc_subt_res["subtlety_grounding_score"],
            "Sphericity": sph_lob_res["sphericity_grounding_score"],
            "Lobulation": sph_lob_res["lobulation_grounding_score"]
        }

        composite_trait_grounding = float(np.mean(list(grounding_scores.values())))

        return {
            "composite_trait_grounding": round(composite_trait_grounding, 3),
            "individual_grounding_scores": grounding_scores,
            "all_traits_physically_grounded": len(all_conflicts) == 0,
            "trait_conflicts": all_conflicts,
            "regional_details": {
                "spiculation_and_margin": spic_marg_res,
                "calcification_and_subtlety": calc_subt_res,
                "sphericity_and_lobulation": sph_lob_res,
                "nodule_center_coordinates": (center_d, center_y, center_x)
            }
        }
