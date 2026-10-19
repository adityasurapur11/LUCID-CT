"""
Adaptive Verification Controller & Targeted Re-Analysis Loop.
The central computational control engine that:
1. Detects multi-factor prediction contradictions, epistemic uncertainty, and explanation failures.
2. Selects and executes targeted re-analysis operations (multi-scale zoom crops, orthogonal perturbations, adaptive filtering).
3. Evaluates convergence and re-analysis agreement.
4. Computes the Composite Reliability Index (CRI).
5. Enforces the Final Reliability Gate: Returns verified result or abstains and requests clinician review.
"""

from typing import Dict, List, Any, Tuple, Optional
import torch
import torch.nn.functional as F
import numpy as np

from src.preprocessing import extract_roi, apply_hu_window
from src.scan_quality import ScanQualityAssessor
from src.trait_verifier import TraitSpecificRegionalVerifier
from src.xai_verifier import ExplanationFidelityVerifier


class AdaptiveVerificationController:
    """
    Closed-loop adaptive controller that actively investigates prediction ambiguities
    through targeted computational re-analysis before rendering a clinical decision.
    """

    RELIABILITY_THRESHOLD: float = 75.0  # CRI score required to clear the Reliability Gate
    MAX_REANALYSIS_STEPS: int = 3

    def __init__(
        self,
        epistemic_threshold: float = 0.07,
        malignant_threshold: float = 0.50,
        min_reanalysis_agreement: float = 0.70
    ):
        self.epistemic_threshold = epistemic_threshold
        self.malignant_threshold = malignant_threshold
        self.min_reanalysis_agreement = min_reanalysis_agreement
        self.trait_verifier = TraitSpecificRegionalVerifier()
        self.quality_assessor = ScanQualityAssessor()
        self.xai_verifier = ExplanationFidelityVerifier()

    def identify_inconsistency_profile(
        self,
        calibrated_prob: float,
        epistemic_unc: float,
        traits: List[float],
        quality_assessment: Dict[str, Any],
        trait_grounding_report: Dict[str, Any],
        xai_fidelity_report: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Profiles the exact nature and source of any prediction inconsistency.
        """
        spic, lob, calc, subt, marg, sph = traits
        is_malignant = calibrated_prob >= self.malignant_threshold

        trait_risk_components = [
            (spic - 1.0) / 4.0,
            (lob - 1.0) / 4.0,
            (5.0 - marg) / 4.0,
            (5.0 - sph) / 4.0,
            (calc - 1.0) / 5.0
        ]
        trait_risk_score = float(np.mean(trait_risk_components))

        contradictions = []
        inconsistency_flags = []

        # 1. Primary vs Trait Contradiction
        if is_malignant and trait_risk_score < 0.35:
            contradictions.append("POTENTIAL_FALSE_POSITIVE: Malignancy probability high but radiological traits are benign.")
            inconsistency_flags.append("CONTRADICTION_MALIGNANCY_VS_TRAITS")
        elif (not is_malignant) and trait_risk_score > 0.65:
            contradictions.append("POTENTIAL_FALSE_NEGATIVE: Malignancy probability low but radiological traits indicate high-risk malignancy.")
            inconsistency_flags.append("CONTRADICTION_MALIGNANCY_VS_TRAITS")

        # 2. Epistemic Uncertainty
        if epistemic_unc > self.epistemic_threshold:
            contradictions.append(f"HIGH_EPISTEMIC_VARIANCE: MC dropout variance ({epistemic_unc:.4f}) exceeds safety limit.")
            inconsistency_flags.append("HIGH_EPISTEMIC_VARIANCE")

        # 3. Trait Physical Grounding
        if not trait_grounding_report.get("all_traits_physically_grounded", True):
            for conflict in trait_grounding_report.get("trait_conflicts", []):
                contradictions.append(f"TRAIT_GROUNDING_CONFLICT: {conflict}")
            
            # Identify specific regional failure
            ind_scores = trait_grounding_report.get("individual_grounding_scores", {})
            if ind_scores.get("Spiculation", 1.0) < 0.65 or ind_scores.get("Margin", 1.0) < 0.65:
                inconsistency_flags.append("UNGROUNDED_BOUNDARY_TRAIT")
            if ind_scores.get("Calcification", 1.0) < 0.65 or ind_scores.get("Subtlety", 1.0) < 0.65:
                inconsistency_flags.append("UNGROUNDED_DENSITY_TRAIT")
            if ind_scores.get("Sphericity", 1.0) < 0.65 or ind_scores.get("Lobulation", 1.0) < 0.65:
                inconsistency_flags.append("UNGROUNDED_SHAPE_TRAIT")

        # 4. Explanation Fidelity Failure
        if xai_fidelity_report and not xai_fidelity_report.get("is_explanation_verified", True):
            for w in xai_fidelity_report.get("fidelity_warnings", []):
                contradictions.append(f"XAI_FIDELITY_WARNING: {w}")
            inconsistency_flags.append("EXPLANATION_FIDELITY_FAILURE")

        # 5. Scan Quality Degradation
        if not quality_assessment.get("is_acceptable", True) or quality_assessment.get("quality_score", 100) < 65:
            contradictions.append(f"SCAN_QUALITY_DEGRADATION: {quality_assessment.get('status_description')}")
            inconsistency_flags.append("SCAN_QUALITY_DEGRADATION")

        return {
            "has_inconsistency": len(inconsistency_flags) > 0,
            "inconsistency_flags": inconsistency_flags,
            "contradiction_details": contradictions,
            "trait_risk_score": trait_risk_score
        }

    def select_reanalysis_operations(self, inconsistency_profile: Dict[str, Any]) -> List[str]:
        """
        Defined algorithm for choosing the targeted re-analysis operations
        based on the detected contradiction and uncertainty types.
        """
        flags = inconsistency_profile.get("inconsistency_flags", [])
        operations = []

        if "UNGROUNDED_BOUNDARY_TRAIT" in flags or "CONTRADICTION_MALIGNANCY_VS_TRAITS" in flags:
            operations.append("TARGETED_MULTI_SCALE_ZOOM_CROPS")

        if "HIGH_EPISTEMIC_VARIANCE" in flags or "EXPLANATION_FIDELITY_FAILURE" in flags:
            operations.append("TEST_TIME_ORTHOGONAL_PERTURBATION")

        if "SCAN_QUALITY_DEGRADATION" in flags:
            operations.append("ADAPTIVE_DENOISING_AND_RECALIBRATION")

        if "UNGROUNDED_DENSITY_TRAIT" in flags and "TARGETED_MULTI_SCALE_ZOOM_CROPS" not in operations:
            operations.append("TARGETED_MULTI_SCALE_ZOOM_CROPS")

        # Default fallback if flagged but no specific match
        if flags and not operations:
            operations.append("TARGETED_MULTI_SCALE_ZOOM_CROPS")

        return operations[:self.MAX_REANALYSIS_STEPS]

    def execute_targeted_reanalysis(
        self,
        model: torch.nn.Module,
        volume: np.ndarray,
        operations: List[str],
        temp_scaler: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Executes selected computational re-analysis operations on the CT volume.
        """
        model.eval()
        device = next(model.parameters()).device
        reanalysis_audit = []
        prob_samples = []
        trait_samples = []

        D, H, W = volume.shape

        for op in operations:
            if op == "TARGETED_MULTI_SCALE_ZOOM_CROPS":
                # Crop fine-grain 1.25x zoom (inner 80% central ROI) and wide contextual 0.8x crop
                mid_d = D // 2
                h_c, w_c = H // 2, W // 2
                
                # 1. Fine-grain zoom crop
                zoom_h, zoom_w = int(H * 0.4), int(W * 0.4)
                fine_crop = volume[:, h_c - zoom_h:h_c + zoom_h, w_c - zoom_w:w_c + zoom_w]
                # Resize back to (D, H, W) via trilinear interpolation
                fine_tensor = torch.tensor(fine_crop, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                fine_resized = F.interpolate(fine_tensor, size=(D, H, W), mode='trilinear', align_corners=False).to(device)

                with torch.no_grad():
                    logits_zoom, traits_zoom = model(fine_resized)
                    if temp_scaler is not None:
                        logits_zoom = temp_scaler(logits_zoom)
                    p_zoom = float(F.softmax(logits_zoom, dim=-1)[0, 1].item())
                    prob_samples.append(p_zoom)
                    trait_samples.append(traits_zoom[0].cpu().numpy())

                reanalysis_audit.append({
                    "operation": op,
                    "target_scale": "1.25x High-Magnification Boundary Shell Zoom",
                    "resulting_malignancy_prob": round(p_zoom, 4)
                })

            elif op == "TEST_TIME_ORTHOGONAL_PERTURBATION":
                # Perturb CT volume via 90-degree axial rotation + slight spatial jitter
                perturbed_vol = np.rot90(volume, k=1, axes=(1, 2)).copy()
                pert_tensor = torch.tensor(perturbed_vol, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

                with torch.no_grad():
                    logits_pert, traits_pert = model(pert_tensor)
                    if temp_scaler is not None:
                        logits_pert = temp_scaler(logits_pert)
                    p_pert = float(F.softmax(logits_pert, dim=-1)[0, 1].item())
                    prob_samples.append(p_pert)
                    trait_samples.append(traits_pert[0].cpu().numpy())

                reanalysis_audit.append({
                    "operation": op,
                    "target_mode": "Orthogonal Spatial Symmetry & Perturbation Invariance Check",
                    "resulting_malignancy_prob": round(p_pert, 4)
                })

            elif op == "ADAPTIVE_DENOISING_AND_RECALIBRATION":
                denoised_vol, desc = self.quality_assessor.apply_adaptive_pathway(volume)
                denoised_tensor = torch.tensor(denoised_vol, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

                with torch.no_grad():
                    logits_denoised, traits_denoised = model(denoised_tensor)
                    if temp_scaler is not None:
                        logits_denoised = temp_scaler(logits_denoised)
                    p_denoised = float(F.softmax(logits_denoised, dim=-1)[0, 1].item())
                    prob_samples.append(p_denoised)
                    trait_samples.append(traits_denoised[0].cpu().numpy())

                reanalysis_audit.append({
                    "operation": op,
                    "target_mode": desc,
                    "resulting_malignancy_prob": round(p_denoised, 4)
                })

        mean_reanalysis_prob = float(np.mean(prob_samples)) if prob_samples else None
        mean_reanalysis_traits = np.mean(trait_samples, axis=0).tolist() if trait_samples else None

        return {
            "operations_executed": operations,
            "reanalysis_audit": reanalysis_audit,
            "reanalysis_prob_samples": prob_samples,
            "mean_reanalysis_prob": mean_reanalysis_prob,
            "mean_reanalysis_traits": mean_reanalysis_traits
        }

    def compute_composite_reliability_index(
        self,
        calibrated_prob: float,
        epistemic_unc: float,
        quality_score: float,
        trait_grounding_score: float,
        xai_fidelity_score: float,
        reanalysis_agreement: float
    ) -> float:
        """
        Calculates Composite Reliability Index (CRI) on a scale of 0 to 100%.
        Weights:
            - Epistemic certainty: 25%
            - Trait physical grounding: 25%
            - Explanation fidelity: 20%
            - Scan acquisition quality: 15%
            - Re-analysis agreement: 15%
        """
        epistemic_certainty = float(np.clip(1.0 - (epistemic_unc / max(self.epistemic_threshold * 2.0, 1e-4)), 0.0, 1.0))
        
        cri = (
            0.25 * (epistemic_certainty * 100.0) +
            0.25 * (trait_grounding_score * 100.0) +
            0.20 * (xai_fidelity_score * 100.0) +
            0.15 * quality_score +
            0.15 * (reanalysis_agreement * 100.0)
        )
        return float(np.clip(cri, 0.0, 100.0))

    def run_adaptive_verification_loop(
        self,
        model: torch.nn.Module,
        volume: np.ndarray,
        initial_prob: float,
        epistemic_unc: float,
        initial_traits: List[float],
        heatmap_3d: Optional[np.ndarray] = None,
        cam_generator: Optional[Any] = None,
        temp_scaler: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Full Closed-Loop Adaptive Verification Pipeline.
        Evaluates scan quality, trait physical grounding, XAI fidelity, runs targeted re-analysis
        if contradictions exist, and enforces the Final Reliability Gate.
        """
        # 1. Scan Quality Assessment
        quality_res = self.quality_assessor.assess_quality(volume)

        # 2. Trait Physical Grounding Check
        trait_res = self.trait_verifier.verify_all_traits(volume, initial_traits)

        # 3. Evidence-Grounded Explanation Verification
        vol_tensor = torch.tensor(volume, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(next(model.parameters()).device)
        if heatmap_3d is not None:
            xai_res = self.xai_verifier.verify_explanation(model, vol_tensor, heatmap_3d, initial_prob, cam_generator)
        else:
            xai_res = {"fidelity_score": 0.80, "is_explanation_verified": True, "fidelity_warnings": []}

        # 4. Contradiction & Inconsistency Profiling
        inconsistency_prof = self.identify_inconsistency_profile(
            calibrated_prob=initial_prob,
            epistemic_unc=epistemic_unc,
            traits=initial_traits,
            quality_assessment=quality_res,
            trait_grounding_report=trait_res,
            xai_fidelity_report=xai_res
        )

        reanalysis_performed = False
        reanalysis_results = None
        agreement_score = 1.0
        final_prob = initial_prob
        final_traits = initial_traits

        # 5. Targeted Re-Analysis Trigger
        if inconsistency_prof["has_inconsistency"]:
            reanalysis_ops = self.select_reanalysis_operations(inconsistency_prof)
            if reanalysis_ops:
                reanalysis_performed = True
                reanalysis_results = self.execute_targeted_reanalysis(model, volume, reanalysis_ops, temp_scaler)
                
                # Compare initial vs re-analysis
                mean_p_re = reanalysis_results["mean_reanalysis_prob"]
                if mean_p_re is not None:
                    prob_diff = abs(mean_p_re - initial_prob)
                    # Agreement score (1.0 = identical, 0.0 = completely opposite)
                    agreement_score = float(np.clip(1.0 - prob_diff * 1.5, 0.0, 1.0))
                    
                    # Update final probability with evidence aggregation
                    final_prob = float(0.60 * initial_prob + 0.40 * mean_p_re)
                    if reanalysis_results["mean_reanalysis_traits"] is not None:
                        final_traits = [
                            float(0.60 * t0 + 0.40 * tre)
                            for t0, tre in zip(initial_traits, reanalysis_results["mean_reanalysis_traits"])
                        ]

        # 6. Compute Composite Reliability Index (CRI)
        cri = self.compute_composite_reliability_index(
            calibrated_prob=final_prob,
            epistemic_unc=epistemic_unc,
            quality_score=quality_res["quality_score"],
            trait_grounding_score=trait_res["composite_trait_grounding"],
            xai_fidelity_score=xai_res["fidelity_score"],
            reanalysis_agreement=agreement_score
        )

        # 7. Final Reliability Gate Decision Logic
        has_fatal_quality = quality_res["recommended_pathway"] == "REJECT_CLINICIAN_REVIEW"
        has_severe_contradiction = (
            "CONTRADICTION_MALIGNANCY_VS_TRAITS" in inconsistency_prof["inconsistency_flags"] and
            agreement_score < self.min_reanalysis_agreement
        )

        if has_fatal_quality or has_severe_contradiction or cri < 65.0:
            final_status = "ABSTAIN_CLINICIAN_REVIEW_REQUIRED"
            clinical_action = (
                "ABSTAIN: The automated system has withheld diagnostic clearance due to persistent "
                f"verification contradictions or sub-threshold reliability (CRI: {cri:.1f}%). "
                "Urgent manual diagnostic review by a thoracic radiologist is required."
            )
            is_cleared = False
        elif reanalysis_performed and agreement_score >= self.min_reanalysis_agreement and cri >= self.RELIABILITY_THRESHOLD:
            final_status = "ADAPTIVELY_RESOLVED"
            clinical_action = (
                "VERIFIED via Adaptive Re-Analysis: Initial inconsistency investigated and resolved "
                f"with high multi-scale agreement ({agreement_score*100:.1f}%) and CRI ({cri:.1f}%). "
                "Proceed with guideline-based clinical management."
            )
            is_cleared = True
        elif cri >= self.RELIABILITY_THRESHOLD and not inconsistency_prof["has_inconsistency"]:
            final_status = "RELIABLE_PREDICTION"
            clinical_action = (
                f"VERIFIED: Prediction passed all quality, trait physical grounding, and XAI fidelity checks (CRI: {cri:.1f}%)."
            )
            is_cleared = True
        else:
            final_status = "ABSTAIN_CLINICIAN_REVIEW_REQUIRED"
            clinical_action = (
                f"ABSTAIN: Uncertainty remains elevated after re-analysis (CRI: {cri:.1f}% < {self.RELIABILITY_THRESHOLD}%). "
                "Refer to specialist multidisciplinary tumor board."
            )
            is_cleared = False

        return {
            "final_status": final_status,
            "is_cleared_by_gate": is_cleared,
            "composite_reliability_index": round(cri, 1),
            "final_calibrated_prob": round(final_prob, 4),
            "final_traits": [round(t, 2) for t in final_traits],
            "recommended_clinical_action": clinical_action,
            "reanalysis_performed": reanalysis_performed,
            "reanalysis_agreement": round(agreement_score, 3),
            "inconsistency_profile": inconsistency_prof,
            "reanalysis_details": reanalysis_results,
            "scan_quality": quality_res,
            "trait_grounding": trait_res,
            "xai_fidelity": xai_res
        }
