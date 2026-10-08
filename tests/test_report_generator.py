import re
from io import BytesIO

import pytest
from PIL import Image

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
    patient_info = {"patient_id": "TEST", "scan_date": "2026-10-09"}
    prediction_results = {
            "calibrated_prob": 0.3,
            "epistemic_uncertainty": 0.02,
            "aleatoric_uncertainty": 0.4,
            "total_uncertainty": 0.5,
            "trait_predictions": [2.0] * 6,
        }
    verification_results = {
            "status": "VERIFIED_MATCH",
            "is_verified": True,
            "final_diagnosis": "Low risk",
            "recommended_action": "Clinical review",
            "trait_breakdown": {
                "Spiculation (1-5)": "2.00",
                "Lobulation (1-5)": "2.00",
                "Calcification (1-6)": "2.00",
                "Subtlety (1-5)": "2.00",
                "Margin (1-5)": "2.00",
                "Sphericity (1-5)": "2.00",
            },
            "conflicts": [],
        }
    report_generator = RadiologicalReportGenerator()
    report_args = {
        "patient_info": patient_info,
        "prediction_results": prediction_results,
        "verification_results": verification_results,
        "lung_rads_info": lung_rads_info,
        "gradcam_img_bytes": b"",
    }

    pdf_bytes = report_generator.generate_pdf_bytes(**report_args)
    image_bytes = report_generator.generate_image_bytes(**report_args)

    assert pdf_bytes.startswith(b"%PDF")
    assert len(re.findall(rb"/Type\s*/Page\b", pdf_bytes)) == 1
    assert image_bytes.startswith(b"\x89PNG\r\n\x1a\n")
    with Image.open(BytesIO(image_bytes)) as report_image:
        assert report_image.size == report_generator._PAGE_SIZE
