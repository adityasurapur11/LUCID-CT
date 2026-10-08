import pytest

from src.report_generator import LungRADSClassifier, RadiologicalReportGenerator


def test_lung_rads_estimate_uses_diameter_and_discloses_assumption():
    result = LungRADSClassifier.classify(
        calibrated_prob=0.7,
        traits=[3.0] * 6,
        epistemic_uncertainty=0.02,
        nodule_radius_mm=5.0,
    )

    assert result["category"] == "4A"
    assert result["nodule_diameter_mm"] == 10.0
    assert result["risk_tier"] == "High"
    assert "assuming a baseline solid nodule" in result["assessment_note"]


def test_lung_rads_estimate_rejects_incomplete_traits():
    with pytest.raises(ValueError, match="six radiological predictions"):
        LungRADSClassifier.classify(
            calibrated_prob=0.5,
            traits=[1.0] * 5,
            epistemic_uncertainty=0.01,
            nodule_radius_mm=5.0,
        )


def test_report_generator_returns_pdf_bytes():
    lung_rads_info = LungRADSClassifier.classify(
        calibrated_prob=0.3,
        traits=[2.0] * 6,
        epistemic_uncertainty=0.02,
        nodule_radius_mm=5.0,
    )
    pdf_bytes = RadiologicalReportGenerator().generate_pdf_bytes(
        patient_info={"patient_id": "TEST", "scan_date": "2026-10-09"},
        prediction_results={
            "calibrated_prob": 0.3,
            "epistemic_uncertainty": 0.02,
        },
        verification_results={
            "status": "VERIFIED_MATCH",
            "final_diagnosis": "Low risk",
            "recommended_action": "Clinical review",
            "conflicts": [],
        },
        lung_rads_info=lung_rads_info,
        gradcam_img_bytes=b"",
    )

    assert pdf_bytes.startswith(b"%PDF")
