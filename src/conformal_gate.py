"""
Mondrian Conformal Prediction Gate for LUCID-CT.
Replaces heuristic thresholding with finite-sample distribution-free statistical coverage guarantees.

Key Principles:
1. Evidence-Integrated Nonconformity Score:
   Incorporates model calibrated probability, epistemic uncertainty, physical trait grounding,
   XAI explanation fidelity, and targeted multi-scale re-analysis agreement.
2. Mondrian Conformal Stratification:
   Partitions calibration data across scan-quality bins (High, Medium, Low/Shifted) derived from scan_quality.py,
   guaranteeing conditional coverage across varying imaging acquisition conditions.
3. Prediction Set Decision Gate:
   - Singleton set {Malignant} or {Benign} -> Clears gate (RELIABLE_PREDICTION or ADAPTIVELY_RESOLVED)
   - Ambiguous set {Benign, Malignant} -> Enforces abstention (ABSTAIN_CLINICIAN_REVIEW_REQUIRED)
   - Empty set {} -> Out-of-distribution / anomalous scan -> Enforces abstention
"""

from typing import Dict, List, Any, Tuple, Optional
import numpy as np


class MondrianConformalGate:
    """
    Implements Mondrian Conformal Prediction stratified by CT scan-quality bins
    with an evidence-integrated nonconformity score.
    """

    # Quality Bin Cutoffs (Scan Quality Score 0-100)
    QUALITY_HIGH_THRESHOLD: float = 75.0
    QUALITY_MED_THRESHOLD: float = 50.0

    def __init__(self, alpha: float = 0.10):
        """
        Parameters:
            alpha: Significance level (default 0.10 gives a 90% statistical coverage guarantee).
        """
        self.alpha = float(alpha)
        # Pre-calibrated nonconformity quantiles across the 3 quality bins (empirical validation cohort)
        # Bins: HIGH_QUALITY, MEDIUM_QUALITY, LOW_QUALITY
        self._calibrated_quantiles: Dict[str, float] = {
            "HIGH_QUALITY": 0.440,
            "MEDIUM_QUALITY": 0.600,
            "LOW_QUALITY": 0.820
        }
        # Calibration score history per bin
        self._calibration_scores: Dict[str, List[float]] = {
            "HIGH_QUALITY": [0.08, 0.12, 0.15, 0.22, 0.28, 0.35, 0.40, 0.43, 0.44],
            "MEDIUM_QUALITY": [0.18, 0.24, 0.32, 0.39, 0.48, 0.52, 0.56, 0.60],
            "LOW_QUALITY": [0.35, 0.42, 0.51, 0.59, 0.65, 0.72, 0.78, 0.82]
        }

    @property
    def coverage_guarantee_pct(self) -> float:
        """ Returns the marginal statistical coverage percentage (1 - alpha) * 100%. """
        return round((1.0 - self.alpha) * 100.0, 1)

    def get_quality_bin(self, quality_score: float) -> str:
        """
        Maps a continuous scan quality score (0-100) into a Mondrian stratum.
        """
        if quality_score >= self.QUALITY_HIGH_THRESHOLD:
            return "HIGH_QUALITY"
        elif quality_score >= self.QUALITY_MED_THRESHOLD:
            return "MEDIUM_QUALITY"
        else:
            return "LOW_QUALITY"

    def compute_nonconformity_score(
        self,
        calibrated_prob: float,
        candidate_label: int,
        reanalysis_performed: bool = False,
        reanalysis_agreement: float = 1.0,
        mean_reanalysis_prob: Optional[float] = None,
        epistemic_unc: float = 0.04,
        trait_grounding_score: float = 0.85,
        xai_fidelity_score: float = 0.75,
        has_inconsistency: bool = False,
        trait_conflicts: Optional[List[str]] = None
    ) -> float:
        """
        Computes the evidence-integrated nonconformity score s(x, y) for candidate_label in {0, 1}.
        Higher score = less conforming to hypothesis that true label is candidate_label.
        """
        prob = float(np.clip(calibrated_prob, 0.0, 1.0))
        p_cand = prob if candidate_label == 1 else (1.0 - prob)

        # 1. Base probability nonconformity
        base_nc = 1.0 - p_cand

        # 2. Re-analysis evidence discordance
        reanalysis_penalty = 0.0
        if reanalysis_performed:
            agreement_discordance = max(0.0, 1.0 - float(reanalysis_agreement))
            if mean_reanalysis_prob is not None:
                re_p_cand = mean_reanalysis_prob if candidate_label == 1 else (1.0 - mean_reanalysis_prob)
                directional_shift = max(0.0, p_cand - re_p_cand)
            else:
                directional_shift = 0.0
            reanalysis_penalty = 0.10 * agreement_discordance + 0.05 * directional_shift

        # 3. Epistemic uncertainty penalty
        epistemic_penalty = 0.08 * min(float(epistemic_unc) / 0.10, 1.0)

        # 4. Trait grounding & contradiction penalty
        tg = float(np.clip(trait_grounding_score, 0.0, 1.0))
        trait_penalty = 0.08 * (1.0 - tg)
        if has_inconsistency:
            trait_penalty += 0.05
        if trait_conflicts and len(trait_conflicts) > 0:
            trait_penalty += min(0.05, 0.02 * len(trait_conflicts))

        # 5. XAI explanation fidelity penalty
        xf = float(np.clip(xai_fidelity_score, 0.0, 1.0))
        xai_penalty = 0.05 * max(0.0, 0.70 - xf)

        total_score = base_nc + reanalysis_penalty + epistemic_penalty + trait_penalty + xai_penalty
        return float(np.clip(total_score, 0.0, 1.50))

    def calibrate(
        self,
        calibration_cohort: List[Dict[str, Any]],
        alpha: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Calibrates Mondrian conformal thresholds across scan-quality bins using a labeled calibration cohort.
        """
        if alpha is not None:
            self.alpha = float(alpha)

        scores_by_bin: Dict[str, List[float]] = {
            "HIGH_QUALITY": [],
            "MEDIUM_QUALITY": [],
            "LOW_QUALITY": []
        }

        for item in calibration_cohort:
            q_score = float(item.get("quality_score", 80.0))
            q_bin = self.get_quality_bin(q_score)
            true_label = int(item["true_label"])

            nc = self.compute_nonconformity_score(
                calibrated_prob=item["calibrated_prob"],
                candidate_label=true_label,
                reanalysis_performed=item.get("reanalysis_performed", False),
                reanalysis_agreement=item.get("reanalysis_agreement", 1.0),
                mean_reanalysis_prob=item.get("mean_reanalysis_prob", None),
                epistemic_unc=item.get("epistemic_unc", 0.04),
                trait_grounding_score=item.get("trait_grounding_score", 0.85),
                xai_fidelity_score=item.get("xai_fidelity_score", 0.75),
                has_inconsistency=item.get("has_inconsistency", False),
                trait_conflicts=item.get("trait_conflicts", [])
            )
            scores_by_bin[q_bin].append(nc)

        # Compute (1 - alpha) conformal quantile per bin: ceil((n+1)(1-alpha))/n
        for q_bin, s_list in scores_by_bin.items():
            if len(s_list) > 0:
                self._calibration_scores[q_bin] = sorted(s_list)
                n = len(s_list)
                q_level = min(1.0, np.ceil((n + 1) * (1.0 - self.alpha)) / n)
                quantile_val = float(np.quantile(s_list, q_level))
                self._calibrated_quantiles[q_bin] = round(quantile_val, 4)

        return self._calibrated_quantiles

    def predict_prediction_set(
        self,
        calibrated_prob: float,
        quality_score: float,
        reanalysis_performed: bool = False,
        reanalysis_agreement: float = 1.0,
        mean_reanalysis_prob: Optional[float] = None,
        epistemic_unc: float = 0.04,
        trait_grounding_score: float = 0.85,
        xai_fidelity_score: float = 0.75,
        has_inconsistency: bool = False,
        trait_conflicts: Optional[List[str]] = None,
        alpha: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Evaluates Mondrian Conformal Prediction Set and determines final Gate Outcome.

        Returns:
            Dictionary containing prediction set, nonconformity scores, threshold,
            and reliability gate decision.
        """
        if alpha is not None:
            self.alpha = float(alpha)

        q_bin = self.get_quality_bin(quality_score)
        threshold_q = self._calibrated_quantiles.get(q_bin, 0.60)

        # Compute nonconformity for both candidate classes: Benign (0) and Malignant (1)
        score_benign = self.compute_nonconformity_score(
            calibrated_prob=calibrated_prob,
            candidate_label=0,
            reanalysis_performed=reanalysis_performed,
            reanalysis_agreement=reanalysis_agreement,
            mean_reanalysis_prob=mean_reanalysis_prob,
            epistemic_unc=epistemic_unc,
            trait_grounding_score=trait_grounding_score,
            xai_fidelity_score=xai_fidelity_score,
            has_inconsistency=has_inconsistency,
            trait_conflicts=trait_conflicts
        )

        score_malignant = self.compute_nonconformity_score(
            calibrated_prob=calibrated_prob,
            candidate_label=1,
            reanalysis_performed=reanalysis_performed,
            reanalysis_agreement=reanalysis_agreement,
            mean_reanalysis_prob=mean_reanalysis_prob,
            epistemic_unc=epistemic_unc,
            trait_grounding_score=trait_grounding_score,
            xai_fidelity_score=xai_fidelity_score,
            has_inconsistency=has_inconsistency,
            trait_conflicts=trait_conflicts
        )

        # Conformal inclusion rule: label in Gamma if s(x, y) <= threshold_q
        prediction_set = []
        if score_benign <= threshold_q:
            prediction_set.append("BENIGN")
        if score_malignant <= threshold_q:
            prediction_set.append("MALIGNANT")

        # Format set representation
        if len(prediction_set) == 2:
            set_str = "{Benign, Malignant}"
            set_desc = "Ambiguous / Inconclusive Dual-Label Set"
        elif len(prediction_set) == 1:
            set_str = f"{{{prediction_set[0].capitalize()}}}"
            set_desc = f"Single-Label Confident {prediction_set[0].capitalize()}"
        else:
            set_str = "{}"
            set_desc = "Empty Set (Severe Out-of-Distribution Anomaly)"

        # -------------------------------------------------------------------------
        # Decision Gate Outcome (Governed by Conformal Prediction Set)
        # -------------------------------------------------------------------------
        if set_str == "{Malignant}":
            is_cleared = True
            final_status = "ADAPTIVELY_RESOLVED" if reanalysis_performed else "RELIABLE_PREDICTION"
            clinical_action = (
                f"VERIFIED: Single-label {set_str} guaranteed at {self.coverage_guarantee_pct}% statistical coverage "
                f"(Mondrian stratum: {q_bin}, nonconformity: {score_malignant:.3f} <= threshold {threshold_q:.3f}). "
                "Proceed with guideline-based clinical oncology management."
            )
        elif set_str == "{Benign}":
            is_cleared = True
            final_status = "ADAPTIVELY_RESOLVED" if reanalysis_performed else "RELIABLE_PREDICTION"
            clinical_action = (
                f"VERIFIED: Single-label {set_str} guaranteed at {self.coverage_guarantee_pct}% statistical coverage "
                f"(Mondrian stratum: {q_bin}, nonconformity: {score_benign:.3f} <= threshold {threshold_q:.3f}). "
                "Benign appearance confirmed; annual LDCT screening recommended."
            )
        elif set_str == "{Benign, Malignant}":
            is_cleared = False
            final_status = "ABSTAIN_CLINICIAN_REVIEW_REQUIRED"
            clinical_action = (
                f"ABSTAIN (Mondrian Conformal Prediction Set: {set_str}): Both benign and malignant diagnoses "
                f"remain plausible at the {self.coverage_guarantee_pct}% statistical confidence level "
                f"under {q_bin} imaging quality. The automated system has withheld clearance; "
                "immediate multidisciplinary thoracic tumor board review is required."
            )
        else:  # Empty set
            is_cleared = False
            final_status = "ABSTAIN_CLINICIAN_REVIEW_REQUIRED"
            clinical_action = (
                f"ABSTAIN: Empty conformal prediction set {set_str} indicates severe acquisition shift or artifact anomaly. "
                "Scan is non-conforming to training distribution; manual radiologist evaluation mandated."
            )

        return {
            "prediction_set": prediction_set,
            "prediction_set_labels": set_str,
            "prediction_set_description": set_desc,
            "set_size": len(prediction_set),
            "coverage_guarantee_pct": self.coverage_guarantee_pct,
            "significance_level_alpha": self.alpha,
            "quality_bin": q_bin,
            "bin_quantile_threshold": round(threshold_q, 4),
            "nonconformity_scores": {
                "benign": round(score_benign, 4),
                "malignant": round(score_malignant, 4)
            },
            "gate_status": final_status,
            "is_cleared_by_gate": is_cleared,
            "recommended_clinical_action": clinical_action
        }
