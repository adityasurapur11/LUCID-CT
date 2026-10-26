"""
Unit tests for the new Adaptive Verification Architecture:
- Scan Quality & Distribution Shift (src/scan_quality.py)
- Trait-Specific Multi-Scale Regional Verification (src/trait_verifier.py)
- Evidence-Grounded XAI Verification (src/xai_verifier.py)
- Adaptive Verification Controller & Re-Analysis Loop (src/adaptive_controller.py)
- Longitudinal Growth & VDT Tracking (src/longitudinal.py)
- 4-Way Comparative Ablation Suite (src/evaluation.py)
- Enhanced Report Generator (src/report_generator.py)
"""

import pytest
import torch
import numpy as np

from src.model import HybridLungCancerModel
from src.uncertainty_calibration import TemperatureScaler
from src.xai import GradCAM3D
from src.scan_quality import ScanQualityAssessor
from src.trait_verifier import TraitSpecificRegionalVerifier
from src.xai_verifier import ExplanationFidelityVerifier
from src.adaptive_controller import AdaptiveVerificationController
from src.longitudinal import LongitudinalNoduleTracker
from src.evaluation import BenchmarkAblationSuite
from src.report_generator import LungRADSClassifier, RadiologicalReportGenerator


@pytest.fixture
def dummy_volume():
    """ Creates a standard normalized 3D CT volume (16, 64, 64) with a central nodule. """
    vol = np.full((16, 64, 64), 0.40, dtype=np.float32)
    # Add central hyperdense nodule
    for d in range(6, 11):
        for y in range(28, 37):
            for x in range(28, 37):
                if (y - 32)**2 + (x - 32)**2 <= 16:
                    vol[d, y, x] = 0.85
    return vol


@pytest.fixture
def model():
    m = HybridLungCancerModel(in_channels=1)
    m.eval()
    return m


def test_scan_quality_assessor(dummy_volume):
    assessor = ScanQualityAssessor()
    
    # 1. Clean volume
    clean_res = assessor.assess_quality(dummy_volume)
    assert "quality_score" in clean_res
    assert clean_res["quality_score"] >= 60.0
    assert clean_res["recommended_pathway"] in ["STANDARD", "HISTOGRAM_RECALIBRATION"]
    assert clean_res["is_acceptable"] is True

    # 2. Noisy volume
    noisy_vol = np.clip(dummy_volume + np.random.normal(0, 0.25, dummy_volume.shape).astype(np.float32), 0.0, 1.0)
    noisy_res = assessor.assess_quality(noisy_vol)
    assert noisy_res["noise_index"] > clean_res["noise_index"]
    assert "DENOISING" in noisy_res["recommended_pathway"] or noisy_res["quality_score"] < clean_res["quality_score"]

    # 3. Apply adaptive pathway
    denoised_vol, desc = assessor.apply_adaptive_pathway(noisy_vol, noisy_res)
    assert denoised_vol.shape == noisy_vol.shape
    assert isinstance(desc, str)


def test_trait_specific_regional_verifier(dummy_volume):
    verifier = TraitSpecificRegionalVerifier()

    # Test all traits: [Spiculation, Lobulation, Calcification, Subtlety, Margin, Sphericity]
    traits = [2.0, 2.0, 4.5, 4.0, 4.0, 4.0]
    report = verifier.verify_all_traits(dummy_volume, traits)

    assert "composite_trait_grounding" in report
    assert 0.0 <= report["composite_trait_grounding"] <= 1.0
    assert "individual_grounding_scores" in report
    assert len(report["individual_grounding_scores"]) == 6
    assert "Spiculation" in report["individual_grounding_scores"]
    assert "Margin" in report["individual_grounding_scores"]
    assert "Calcification" in report["individual_grounding_scores"]


def test_xai_explanation_fidelity(dummy_volume, model):
    target_layer = model.block2.conv
    cam_gen = GradCAM3D(model, target_layer)
    vol_tensor = torch.tensor(dummy_volume, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    heatmap = cam_gen.generate_heatmap(vol_tensor, target_class=1)

    verifier = ExplanationFidelityVerifier()
    res = verifier.verify_explanation(
        model=model,
        input_tensor=vol_tensor,
        heatmap_3d=heatmap,
        base_prob=0.70,
        cam_generator=cam_gen
    )

    assert "fidelity_score" in res
    assert 0.0 <= res["fidelity_score"] <= 1.0
    assert "deletion_test" in res
    assert "insertion_test" in res
    assert "stability_test" in res
    assert "anatomical_plausibility" in res


def test_adaptive_verification_controller(dummy_volume, model):
    controller = AdaptiveVerificationController()
    scaler = TemperatureScaler(temperature=1.2)
    cam_gen = GradCAM3D(model, model.block2.conv)
    vol_tensor = torch.tensor(dummy_volume, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    heatmap = cam_gen.generate_heatmap(vol_tensor, target_class=1)

    # 1. Normal case
    traits = [2.0, 2.0, 5.0, 4.0, 4.0, 4.0]
    res = controller.run_adaptive_verification_loop(
        model=model,
        volume=dummy_volume,
        initial_prob=0.20,
        epistemic_unc=0.03,
        initial_traits=traits,
        heatmap_3d=heatmap,
        cam_generator=cam_gen,
        temp_scaler=scaler
    )

    assert "final_status" in res
    assert res["final_status"] in ["RELIABLE_PREDICTION", "ADAPTIVELY_RESOLVED", "ABSTAIN_CLINICIAN_REVIEW_REQUIRED"]
    assert "composite_reliability_index" in res
    assert 0.0 <= res["composite_reliability_index"] <= 100.0
    assert "reanalysis_performed" in res
    assert "recommended_clinical_action" in res

    assert "conformal_prediction" in res
    assert "prediction_set" in res["conformal_prediction"]
    assert res["conformal_prediction"]["prediction_set_labels"] in ["{Malignant}", "{Benign}", "{Benign, Malignant}", "{}"]
    assert 0.0 < res["conformal_prediction"]["coverage_guarantee_pct"] <= 100.0

    # 2. Severe contradiction case (High cancer prob, but benign traits) -> Triggers re-analysis
    contradictory_traits = [1.0, 1.0, 1.0, 5.0, 5.0, 5.0]
    res_contra = controller.run_adaptive_verification_loop(
        model=model,
        volume=dummy_volume,
        initial_prob=0.85,
        epistemic_unc=0.09,
        initial_traits=contradictory_traits,
        heatmap_3d=heatmap,
        cam_generator=cam_gen,
        temp_scaler=scaler
    )
    assert res_contra["reanalysis_performed"] is True
    assert "inconsistency_profile" in res_contra
    assert "conformal_prediction" in res_contra


def test_mondrian_conformal_gate():
    from src.conformal_gate import MondrianConformalGate
    gate = MondrianConformalGate(alpha=0.10)
    assert gate.coverage_guarantee_pct == 90.0

    # 1. Quality binning test
    assert gate.get_quality_bin(85.0) == "HIGH_QUALITY"
    assert gate.get_quality_bin(65.0) == "MEDIUM_QUALITY"
    assert gate.get_quality_bin(40.0) == "LOW_QUALITY"

    # 2. Confident clean high-quality scan -> singleton set {Malignant} -> clears gate
    res_clean_mal = gate.predict_prediction_set(
        calibrated_prob=0.92,
        quality_score=88.0,
        reanalysis_performed=False,
        reanalysis_agreement=1.0,
        epistemic_unc=0.02,
        trait_grounding_score=0.92,
        xai_fidelity_score=0.85,
        has_inconsistency=False
    )
    assert res_clean_mal["prediction_set_labels"] == "{Malignant}"
    assert res_clean_mal["is_cleared_by_gate"] is True
    assert res_clean_mal["gate_status"] == "RELIABLE_PREDICTION"

    # 3. Confident clean high-quality scan -> singleton set {Benign} -> clears gate
    res_clean_ben = gate.predict_prediction_set(
        calibrated_prob=0.08,
        quality_score=85.0,
        reanalysis_performed=False,
        reanalysis_agreement=1.0,
        epistemic_unc=0.02,
        trait_grounding_score=0.90,
        xai_fidelity_score=0.80,
        has_inconsistency=False
    )
    assert res_clean_ben["prediction_set_labels"] == "{Benign}"
    assert res_clean_ben["is_cleared_by_gate"] is True
    assert res_clean_ben["gate_status"] == "RELIABLE_PREDICTION"

    # 4. Noisy / low-quality / contradictory scan -> wider set {Benign, Malignant} -> enforces ABSTAIN
    res_noisy_contra = gate.predict_prediction_set(
        calibrated_prob=0.55,
        quality_score=38.0,
        reanalysis_performed=True,
        reanalysis_agreement=0.50,
        epistemic_unc=0.09,
        trait_grounding_score=0.45,
        xai_fidelity_score=0.40,
        has_inconsistency=True,
        trait_conflicts=["Spiculation contradiction"]
    )
    assert res_noisy_contra["prediction_set_labels"] == "{Benign, Malignant}"
    assert res_noisy_contra["is_cleared_by_gate"] is False
    assert res_noisy_contra["gate_status"] == "ABSTAIN_CLINICIAN_REVIEW_REQUIRED"


def test_longitudinal_nodule_tracker(dummy_volume):
    tracker = LongitudinalNoduleTracker()

    # 1. Doubling time calculation
    v1 = tracker.calculate_volume_from_diameter(6.0)
    v2 = tracker.calculate_volume_from_diameter(9.0)
    vdt = tracker.calculate_volume_doubling_time(v1, v2, time_elapsed_days=180.0)
    assert vdt is not None
    assert vdt > 0

    # 2. Temporal progression assessment
    prog = tracker.assess_temporal_progression(
        baseline_diameter_mm=6.0,
        followup_diameter_mm=9.5,
        time_elapsed_days=150.0,
        baseline_malignancy_prob=0.30,
        followup_malignancy_prob=0.85
    )
    assert prog["trajectory"] == "RAPID_GROWTH"
    assert "Aggressive" in prog["risk_tier"]

    # 3. Synthetic follow-up generator
    fol_vol, d1, d2 = tracker.generate_synthetic_followup(dummy_volume, time_elapsed_days=180.0, growth_mode="RAPID_GROWTH")
    assert fol_vol.shape == dummy_volume.shape
    assert d2 > d1


def test_benchmark_ablation_suite(model):
    scaler = TemperatureScaler(temperature=1.2)
    suite = BenchmarkAblationSuite(model=model, temp_scaler=scaler)
    
    # Run with small cohort of 4 samples for fast unit testing
    cohort = suite.generate_evaluation_cohort(num_samples=4)
    assert len(cohort) == 4
    
    bench_res = suite.run_benchmark(cohort=cohort)
    assert "comparison_table" in bench_res
    assert "confusion_matrices" in bench_res
    table = bench_res["comparison_table"]

    for exp_key in ["Experiment_A_Baseline", "Experiment_B_RuleBased", "Experiment_C_AdaptiveReanalysis", "Experiment_D_FullSystem"]:
        assert exp_key in table
        assert "AUROC" in table[exp_key]
        assert "Sensitivity" in table[exp_key]
        assert "Selective_Error_Rate_Pct" in table[exp_key]
        assert "Accuracy" in table[exp_key]
        assert "Precision" in table[exp_key]
        assert "F1_Score" in table[exp_key]
        assert "confusion_matrix" in table[exp_key]
        assert "tp" in table[exp_key]["confusion_matrix"]
        assert "fn" in table[exp_key]["confusion_matrix"]


def test_compute_confusion_matrix_standalone():
    from src.evaluation import compute_confusion_matrix
    y_true = np.array([1, 1, 0, 0, 1, 0])
    y_pred = np.array([1, 0, 0, 1, 1, 0])
    cm = compute_confusion_matrix(y_true, y_pred, threshold=0.50)
    assert cm["tp"] == 2
    assert cm["fn"] == 1
    assert cm["fp"] == 1
    assert cm["tn"] == 2
    assert cm["total"] == 6
    assert cm["sensitivity"] == round(2 / 3, 4)
    assert cm["specificity"] == round(2 / 3, 4)
    assert cm["accuracy"] == round(4 / 6, 4)


def test_enhanced_report_generation(dummy_volume):
    report_gen = RadiologicalReportGenerator()
    lung_rads_info = LungRADSClassifier.classify(
        calibrated_prob=0.75,
        traits=[3.5, 3.2, 5.0, 3.8, 2.5, 3.0],
        epistemic_uncertainty=0.04,
        nodule_radius_mm=5.0
    )

    patient_info = {
        "patient_id": "TEST-PATIENT-001",
        "scan_date": "2026-10-19",
        "scan_format": "Synthetic Test Scan",
        "slice_count": 16
    }

    pred_res = {
        "calibrated_prob": 0.75,
        "epistemic_uncertainty": 0.04,
        "aleatoric_uncertainty": 0.15,
        "total_uncertainty": 0.25
    }

    verif_res = {
        "status": "VERIFIED_MATCH",
        "is_verified": True,
        "final_diagnosis": "High Risk Malignant Nodule",
        "recommended_action": "Urgent biopsy",
        "trait_breakdown": {
            "Spiculation (1-5)": "3.50",
            "Lobulation (1-5)": "3.20",
            "Calcification (1-6)": "5.00",
            "Subtlety (1-5)": "3.80",
            "Margin (1-5)": "2.50",
            "Sphericity (1-5)": "3.00"
        }
    }

    adaptive_res = {
        "final_status": "ADAPTIVELY_RESOLVED",
        "is_cleared_by_gate": True,
        "composite_reliability_index": 82.5,
        "reanalysis_performed": True,
        "reanalysis_agreement": 0.88,
        "reanalysis_details": {
            "reanalysis_audit": [{"operation": "TARGETED_MULTI_SCALE_ZOOM_CROPS", "target_scale": "1.25x Boundary Shell Zoom"}]
        },
        "scan_quality": {"quality_score": 92.0},
        "xai_fidelity": {"fidelity_score": 0.78},
        "trait_grounding": {"composite_trait_grounding": 0.84}
    }

    # Generate PDF bytes
    pdf_bytes = report_gen.generate_pdf_bytes(
        patient_info=patient_info,
        prediction_results=pred_res,
        verification_results=verif_res,
        lung_rads_info=lung_rads_info,
        gradcam_img_bytes=b"",
        adaptive_results=adaptive_res
    )
    assert len(pdf_bytes) > 1000
    assert pdf_bytes.startswith(b"%PDF")

    # Generate Image bytes
    img_bytes = report_gen.generate_image_bytes(
        patient_info=patient_info,
        prediction_results=pred_res,
        verification_results=verif_res,
        lung_rads_info=lung_rads_info,
        gradcam_img_bytes=b"",
        adaptive_results=adaptive_res
    )
    assert len(img_bytes) > 1000
