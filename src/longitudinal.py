"""
Longitudinal Consistency Verification Module.
Tracks pulmonary nodule evolution across sequential temporal CT scans (Baseline T1 vs Follow-up T2).
Calculates Volumetric Growth, Volume Doubling Time (VDT in days),
detects measurement quality issues, and assigns clinical risk tiers per Fleischner & BTS guidelines.
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np


class LongitudinalNoduleTracker:
    """
    Analyzes temporal consistency, volumetric doubling time, and morphologic progression
    between baseline and follow-up thoracic CT scans.
    """

    # BTS & Fleischner Guideline thresholds
    RAPID_GROWTH_VDT_DAYS: float = 400.0       # VDT < 400 days indicates rapid aggressive malignant growth
    INDETERMINATE_GROWTH_VDT_DAYS: float = 600.0 # 400 - 600 days indicates intermediate growth
    MIN_SIGNIFICANT_VOLUME_CHANGE_PCT: float = 25.0  # <25% volume change is within scanner measurement noise

    @staticmethod
    def calculate_volume_from_diameter(diameter_mm: float) -> float:
        """ Calculates spherical volume in mm^3: V = (4/3) * pi * (d/2)^3 """
        radius = max(diameter_mm / 2.0, 0.1)
        return float((4.0 / 3.0) * np.pi * (radius ** 3))

    @staticmethod
    def calculate_diameter_from_volume(volume_mm3: float) -> float:
        """ Calculates effective spherical diameter in mm: d = 2 * (3V / 4pi)^(1/3) """
        vol = max(volume_mm3, 1e-3)
        radius = (3.0 * vol / (4.0 * np.pi)) ** (1.0 / 3.0)
        return float(2.0 * radius)

    def calculate_volume_doubling_time(
        self,
        v1_mm3: float,
        v2_mm3: float,
        time_elapsed_days: float
    ) -> Optional[float]:
        """
        Calculates Volume Doubling Time (VDT) using the Schwartz equation:
        VDT = (ln(2) * delta_t) / ln(V2 / V1)
        Returns VDT in days, or None if volume decreased or did not change.
        """
        if v1_mm3 <= 0 or v2_mm3 <= 0 or time_elapsed_days <= 0:
            return None
        
        ratio = v2_mm3 / v1_mm3
        if abs(ratio - 1.0) < 1e-4 or ratio <= 0:
            return None
        
        log_ratio = np.log(ratio)
        if log_ratio <= 0:
            return None  # Volume decreased or static

        vdt = (np.log(2.0) * time_elapsed_days) / log_ratio
        return float(vdt)

    def assess_temporal_progression(
        self,
        baseline_diameter_mm: float,
        followup_diameter_mm: float,
        time_elapsed_days: float,
        baseline_malignancy_prob: float,
        followup_malignancy_prob: float,
        slice_thickness_t1_mm: float = 1.25,
        slice_thickness_t2_mm: float = 1.25
    ) -> Dict[str, Any]:
        """
        Performs comprehensive longitudinal consistency analysis.
        """
        v1 = self.calculate_volume_from_diameter(baseline_diameter_mm)
        v2 = self.calculate_volume_from_diameter(followup_diameter_mm)
        
        vol_change_pct = ((v2 - v1) / v1) * 100.0
        diameter_change_mm = followup_diameter_mm - baseline_diameter_mm

        # Check acquisition consistency
        thickness_diff = abs(slice_thickness_t1_mm - slice_thickness_t2_mm)
        acquisition_reliable = thickness_diff <= 1.0

        vdt_days = self.calculate_volume_doubling_time(v1, v2, time_elapsed_days)

        # Risk Classification
        if vol_change_pct < -self.MIN_SIGNIFICANT_VOLUME_CHANGE_PCT:
            trajectory = "REGRESSION"
            risk_tier = "Very Low Risk (Resolving / Likely Infectious or Inflammatory)"
            clinical_rec = "Nodule decreased in volume by >25%. Suggests resolved infection or hemorrhage. Optional confirmation in 12 months."
        elif abs(vol_change_pct) <= self.MIN_SIGNIFICANT_VOLUME_CHANGE_PCT:
            trajectory = "STABLE"
            risk_tier = "Low Risk (Indolent / Stable)"
            clinical_rec = "Volume change is within measurement variability (<25%). Continue routine screening per Lung-RADS interval."
        elif vdt_days is not None and vdt_days < self.RAPID_GROWTH_VDT_DAYS:
            trajectory = "RAPID_GROWTH"
            risk_tier = "High Risk - Aggressive Malignancy Progression"
            clinical_rec = (
                f"Rapid volumetric growth confirmed (VDT: {vdt_days:.1f} days < {self.RAPID_GROWTH_VDT_DAYS:.0f} days). "
                "Immediate multidisciplinary oncology referral and tissue biopsy recommended."
            )
        elif vdt_days is not None and vdt_days <= self.INDETERMINATE_GROWTH_VDT_DAYS:
            trajectory = "INDETERMINATE_GROWTH"
            risk_tier = "Intermediate Risk (Indeterminate Growth)"
            clinical_rec = (
                f"Indeterminate doubling time (VDT: {vdt_days:.1f} days). "
                "Consider PET-CT imaging or short-interval 3-month CT surveillance."
            )
        else:
            trajectory = "SLOW_GROWTH"
            risk_tier = "Low-to-Intermediate Risk (Indolent Growth)"
            clinical_rec = (
                f"Slow indolent progression (VDT: {vdt_days:.1f} days if calculated). "
                "Consider atypical adenomatous hyperplasia or low-grade adenocarcinoma. Follow up in 6 months."
            )

        # Consistency between AI Model Probability Shift and Geometric Growth
        prob_shift = followup_malignancy_prob - baseline_malignancy_prob
        is_temporally_consistent = True
        temporal_warning = None

        if trajectory == "RAPID_GROWTH" and prob_shift < -0.15:
            is_temporally_consistent = False
            temporal_warning = (
                f"Temporal Contradiction: Nodule grew rapidly ({vol_change_pct:.1f}% volume increase), "
                f"yet model malignancy probability unexpectedly decreased ({baseline_malignancy_prob*100:.1f}% -> {followup_malignancy_prob*100:.1f}%)."
            )
        elif trajectory == "REGRESSION" and prob_shift > 0.25:
            is_temporally_consistent = False
            temporal_warning = (
                f"Temporal Contradiction: Nodule regressed significantly ({vol_change_pct:.1f}%), "
                f"yet model malignancy probability spiked ({baseline_malignancy_prob*100:.1f}% -> {followup_malignancy_prob*100:.1f}%)."
            )

        return {
            "trajectory": trajectory,
            "risk_tier": risk_tier,
            "volume_t1_mm3": round(v1, 1),
            "volume_t2_mm3": round(v2, 1),
            "volume_change_pct": round(vol_change_pct, 1),
            "diameter_change_mm": round(diameter_change_mm, 2),
            "time_elapsed_days": time_elapsed_days,
            "volume_doubling_time_days": round(vdt_days, 1) if vdt_days is not None else None,
            "clinical_recommendation": clinical_rec,
            "acquisition_consistency": {
                "is_reliable": acquisition_reliable,
                "slice_thickness_diff_mm": round(thickness_diff, 2),
                "note": "Reliable acquisition match" if acquisition_reliable else "Warning: Slice thickness differences may distort volumetric estimation."
            },
            "is_temporally_consistent": is_temporally_consistent,
            "temporal_contradiction_warning": temporal_warning
        }

    @staticmethod
    def generate_synthetic_followup(
        baseline_volume: np.ndarray,
        time_elapsed_days: float = 180.0,
        growth_mode: str = "RAPID_GROWTH"
    ) -> Tuple[np.ndarray, float, float]:
        """
        Generates a paired synthetic follow-up volume for demonstration/testing.
        Returns:
            - followup_volume: (D, H, W) numpy array
            - baseline_diameter_mm
            - followup_diameter_mm
        """
        followup = baseline_volume.copy()
        D, H, W = followup.shape
        mid_d = D // 2
        h_c, w_c = H // 2, W // 2

        base_diam = 7.0

        if growth_mode == "RAPID_GROWTH":
            followup_diam = 11.2  # Significant expansion
            # Dilate/amplify center nodule
            r = 10
            for d in range(max(0, mid_d - 4), min(D, mid_d + 5)):
                sl = followup[d]
                yy, xx = np.ogrid[:H, :W]
                mask = (yy - h_c)**2 + (xx - w_c)**2 <= r**2
                sl[mask] = np.clip(sl[mask] + 0.35, 0.0, 1.0)
                followup[d] = sl
        elif growth_mode == "REGRESSION":
            followup_diam = 4.0
            r = 10
            for d in range(max(0, mid_d - 4), min(D, mid_d + 5)):
                sl = followup[d]
                yy, xx = np.ogrid[:H, :W]
                mask = (yy - h_c)**2 + (xx - w_c)**2 <= r**2
                sl[mask] = np.clip(sl[mask] - 0.25, 0.0, 1.0)
                followup[d] = sl
        else:  # STABLE
            followup_diam = 7.2

        return followup, base_diam, followup_diam
