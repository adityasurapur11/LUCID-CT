"""
Comparative Benchmark & Ablation Evaluation Engine.
Implements the 4-way comparative evaluation matrix recommended for research & patent validation:
- Experiment A: Baseline Model (Standard 3D CNN-Transformer)
- Experiment B: Baseline Model + Current Passive Rule-Based Verifier
- Experiment C: Baseline Model + Adaptive Re-Analysis Controller
- Experiment D: Full System (Scan Quality + Trait Grounding + XAI Fidelity + Adaptive Re-Analysis + Reliability Gating)
"""

from typing import Dict, List, Any, Tuple, Optional
import time
import torch
import torch.nn.functional as F
import numpy as np

from src.dataset import SyntheticLIDCDataset
from src.model import HybridLungCancerModel
from src.uncertainty_calibration import TemperatureScaler, compute_expected_calibration_error
from src.self_verification import SelfVerificationEngine
from src.adaptive_controller import AdaptiveVerificationController
from src.xai import GradCAM3D


def compute_auroc_approx(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """ Computes Wilcoxon-Mann-Whitney trapezoidal AUROC approximation without sklearn dependency. """
    pos = y_scores[y_true == 1]
    neg = y_scores[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    
    # Count pairs where pos > neg
    n_pos = len(pos)
    n_neg = len(neg)
    greater = sum(np.sum(p > neg) + 0.5 * np.sum(p == neg) for p in pos)
    return float(greater / (n_pos * n_neg))


def compute_brier_score(y_true: np.ndarray, y_probs: np.ndarray) -> float:
    """ Calculates Brier Score: mean squared error between probabilities and binary labels. """
    return float(np.mean((y_probs - y_true) ** 2))


class BenchmarkAblationSuite:
    """
    Executes Experiments A, B, C, D across an evaluation cohort of standard,
    shifted, and contradictory lung nodule scans.
    """

    def __init__(self, model: HybridLungCancerModel, temp_scaler: Optional[TemperatureScaler] = None):
        self.model = model
        self.temp_scaler = temp_scaler
        self.rule_verifier = SelfVerificationEngine()
        self.adaptive_controller = AdaptiveVerificationController()

    def generate_evaluation_cohort(self, num_samples: int = 24) -> List[Dict[str, Any]]:
        """
        Creates a rigorous evaluation cohort containing:
        - Clean standard synthetic LIDC cases (benign & malignant)
        - Scan-quality shifted / noisy cases
        - Contradictory edge cases (e.g. malignant core with benign margin or faint ground-glass)
        """
        dataset = SyntheticLIDCDataset(num_samples=num_samples, depth=16, height=64, width=64, seed=777)
        cohort = []

        for idx in range(len(dataset)):
            vol_tensor, label_tensor, traits_tensor = dataset[idx]
            vol_np = vol_tensor.squeeze(0).numpy().copy()
            label = int(label_tensor.item())
            traits = traits_tensor.numpy().tolist()

            case_type = "STANDARD"
            # Introduce controlled shifts for specific subsets
            if idx % 4 == 1:
                # Noisy / acquisition shifted scan
                noise = np.random.normal(0, 0.12, vol_np.shape).astype(np.float32)
                vol_np = np.clip(vol_np + noise, 0.0, 1.0)
                case_type = "NOISY_SHIFTED"
            elif idx % 4 == 2:
                # Contradictory edge case: invert traits relative to label to test conflict detection
                if label == 1:
                    traits = [1.2, 1.1, 1.0, 4.8, 4.9, 4.7]  # Malignant label but benign traits
                else:
                    traits = [4.8, 4.5, 5.8, 1.5, 1.2, 1.5]  # Benign label but malignant traits
                case_type = "CONTRADICTORY_TRAIT"

            cohort.append({
                "id": f"PATIENT_EVAL_{idx+1:03d}",
                "volume": vol_np,
                "true_label": label,
                "initial_traits": traits,
                "case_type": case_type
            })

        return cohort

    def run_benchmark(self, cohort: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Executes the 4-way ablation benchmark and outputs quantitative comparison metrics.
        """
        if cohort is None:
            cohort = self.generate_evaluation_cohort(num_samples=20)

        device = next(self.model.parameters()).device
        self.model.eval()

        results = {
            "Experiment_A_Baseline": {"probs": [], "preds": [], "accepted": [], "labels": [], "latencies": [], "flagged_flawed": []},
            "Experiment_B_RuleBased": {"probs": [], "preds": [], "accepted": [], "labels": [], "latencies": [], "flagged_flawed": []},
            "Experiment_C_AdaptiveReanalysis": {"probs": [], "preds": [], "accepted": [], "labels": [], "latencies": [], "flagged_flawed": []},
            "Experiment_D_FullSystem": {"probs": [], "preds": [], "accepted": [], "labels": [], "latencies": [], "flagged_flawed": []}
        }

        flawed_ground_truth = [1 if c["case_type"] != "STANDARD" else 0 for c in cohort]

        for case in cohort:
            vol = case["volume"]
            label = case["true_label"]
            traits = case["initial_traits"]
            D, H, W = vol.shape

            vol_tensor = torch.tensor(vol, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

            # --- EXPERIMENT A: Baseline Model ---
            t0 = time.perf_counter()
            with torch.no_grad():
                logits, pred_traits = self.model(vol_tensor)
                if self.temp_scaler is not None:
                    logits = self.temp_scaler(logits)
                prob_a = float(F.softmax(logits, dim=-1)[0, 1].item())
            lat_a = (time.perf_counter() - t0) * 1000.0
            pred_a = 1 if prob_a >= 0.50 else 0
            results["Experiment_A_Baseline"]["probs"].append(prob_a)
            results["Experiment_A_Baseline"]["preds"].append(pred_a)
            results["Experiment_A_Baseline"]["accepted"].append(True)  # Accepts 100%
            results["Experiment_A_Baseline"]["labels"].append(label)
            results["Experiment_A_Baseline"]["latencies"].append(lat_a)
            results["Experiment_A_Baseline"]["flagged_flawed"].append(0)

            # --- EXPERIMENT B: Baseline + Current Rule-Based Verifier ---
            t0 = time.perf_counter()
            verif_b = self.rule_verifier.verify_prediction(
                calibrated_prob=prob_a,
                epistemic_uncertainty=0.04,
                traits=traits
            )
            lat_b = (time.perf_counter() - t0) * 1000.0 + lat_a
            # Passive verifier flags conflicts as warning/inconclusive
            is_accepted_b = verif_b["status"] == "VERIFIED_MATCH"
            is_flagged_b = 1 if not is_accepted_b else 0
            results["Experiment_B_RuleBased"]["probs"].append(prob_a)
            results["Experiment_B_RuleBased"]["preds"].append(pred_a)
            results["Experiment_B_RuleBased"]["accepted"].append(is_accepted_b)
            results["Experiment_B_RuleBased"]["labels"].append(label)
            results["Experiment_B_RuleBased"]["latencies"].append(lat_b)
            results["Experiment_B_RuleBased"]["flagged_flawed"].append(is_flagged_b)

            # --- EXPERIMENT C: Baseline + Adaptive Re-Analysis Controller ---
            t0 = time.perf_counter()
            # Adaptive controller runs targeted multi-scale zoom if contradiction detected
            incons_c = self.adaptive_controller.identify_inconsistency_profile(
                calibrated_prob=prob_a,
                epistemic_unc=0.04,
                traits=traits,
                quality_assessment={"is_acceptable": True, "quality_score": 90, "recommended_pathway": "STANDARD", "status_description": "Normal"},
                trait_grounding_report={"all_traits_physically_grounded": True, "composite_trait_grounding": 0.85, "trait_conflicts": []}
            )
            prob_c = prob_a
            is_accepted_c = True
            if incons_c["has_inconsistency"]:
                re_ops = self.adaptive_controller.select_reanalysis_operations(incons_c)
                re_res = self.adaptive_controller.execute_targeted_reanalysis(self.model, vol, re_ops, self.temp_scaler)
                if re_res["mean_reanalysis_prob"] is not None:
                    prob_c = float(0.60 * prob_a + 0.40 * re_res["mean_reanalysis_prob"])
                agreement = 1.0 - abs(prob_c - prob_a)
                is_accepted_c = agreement >= 0.70
            lat_c = (time.perf_counter() - t0) * 1000.0 + lat_a
            pred_c = 1 if prob_c >= 0.50 else 0
            results["Experiment_C_AdaptiveReanalysis"]["probs"].append(prob_c)
            results["Experiment_C_AdaptiveReanalysis"]["preds"].append(pred_c)
            results["Experiment_C_AdaptiveReanalysis"]["accepted"].append(is_accepted_c)
            results["Experiment_C_AdaptiveReanalysis"]["labels"].append(label)
            results["Experiment_C_AdaptiveReanalysis"]["latencies"].append(lat_c)
            results["Experiment_C_AdaptiveReanalysis"]["flagged_flawed"].append(1 if not is_accepted_c else 0)

            # --- EXPERIMENT D: Full System (Quality + Trait Grounding + XAI + Adaptive Controller + Reliability Gate) ---
            t0 = time.perf_counter()
            cam_gen = GradCAM3D(self.model, self.model.block2.conv)
            heatmap = cam_gen.generate_heatmap(vol_tensor, target_class=1)
            full_res = self.adaptive_controller.run_adaptive_verification_loop(
                model=self.model,
                volume=vol,
                initial_prob=prob_a,
                epistemic_unc=0.04,
                initial_traits=traits,
                heatmap_3d=heatmap,
                cam_generator=cam_gen,
                temp_scaler=self.temp_scaler
            )
            lat_d = (time.perf_counter() - t0) * 1000.0 + lat_a
            prob_d = full_res["final_calibrated_prob"]
            pred_d = 1 if prob_d >= 0.50 else 0
            is_accepted_d = full_res["is_cleared_by_gate"]
            results["Experiment_D_FullSystem"]["probs"].append(prob_d)
            results["Experiment_D_FullSystem"]["preds"].append(pred_d)
            results["Experiment_D_FullSystem"]["accepted"].append(is_accepted_d)
            results["Experiment_D_FullSystem"]["labels"].append(label)
            results["Experiment_D_FullSystem"]["latencies"].append(lat_d)
            results["Experiment_D_FullSystem"]["flagged_flawed"].append(1 if not is_accepted_d else 0)

        # Compute Summary Metrics Table
        comparison_table = {}
        flawed_gt = np.array(flawed_ground_truth)

        for exp_name, data in results.items():
            probs = np.array(data["probs"])
            preds = np.array(data["preds"])
            labels = np.array(data["labels"])
            accepted = np.array(data["accepted"])
            flagged = np.array(data["flagged_flawed"])

            # 1. Classification Metrics
            tp = int(np.sum((preds == 1) & (labels == 1)))
            tn = int(np.sum((preds == 0) & (labels == 0)))
            fp = int(np.sum((preds == 1) & (labels == 0)))
            fn = int(np.sum((preds == 0) & (labels == 1)))

            sensitivity = float(tp / max(tp + fn, 1))
            specificity = float(tn / max(tn + fp, 1))
            fnr = float(fn / max(tp + fn, 1))
            auroc = compute_auroc_approx(labels, probs)

            # 2. Calibration
            prob_pairs = np.stack([1.0 - probs, probs], axis=1)
            ece = compute_expected_calibration_error(prob_pairs, labels, n_bins=10)
            brier = compute_brier_score(labels, probs)

            # 3. Selective Prediction Metrics
            referral_rate = float(np.mean(~accepted) * 100.0)
            if np.any(accepted):
                acc_labels = labels[accepted]
                acc_preds = preds[accepted]
                selective_error = float(np.mean(acc_labels != acc_preds) * 100.0)
            else:
                selective_error = 0.0

            # 4. Flawed Case Detection Precision & Recall
            verif_tp = int(np.sum((flagged == 1) & (flawed_gt == 1)))
            verif_fp = int(np.sum((flagged == 1) & (flawed_gt == 0)))
            verif_fn = int(np.sum((flagged == 0) & (flawed_gt == 1)))
            verif_prec = float(verif_tp / max(verif_tp + verif_fp, 1))
            verif_rec = float(verif_tp / max(verif_tp + verif_fn, 1))

            avg_latency = float(np.mean(data["latencies"]))

            comparison_table[exp_name] = {
                "AUROC": round(auroc, 3),
                "Sensitivity": round(sensitivity, 3),
                "Specificity": round(specificity, 3),
                "False_Negative_Rate": round(fnr, 3),
                "ECE": round(ece, 4),
                "Brier_Score": round(brier, 4),
                "Selective_Error_Rate_Pct": round(selective_error, 1),
                "Referral_Rate_Pct": round(referral_rate, 1),
                "Verification_Precision": round(verif_prec, 3),
                "Verification_Recall": round(verif_rec, 3),
                "Avg_Latency_ms": round(avg_latency, 1)
            }

        return {
            "comparison_table": comparison_table,
            "total_evaluated_scans": len(cohort),
            "num_flawed_test_cases": int(np.sum(flawed_gt)),
            "raw_experiment_data": results
        }
