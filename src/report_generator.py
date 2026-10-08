"""Lung-RADS decision-support classification and clinical PDF report generation."""

from __future__ import annotations

from io import BytesIO
from math import isfinite
from typing import Any
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (
    Image,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


class LungRADSClassifier:
    """Provide a size-based screening category alongside the model risk estimate.

    Nodule type, growth, and prior examinations are not inputs to this app.
    Categories therefore assume a baseline solid nodule and are estimates,
    not an official Lung-RADS assessment.
    """

    @staticmethod
    def classify(
        calibrated_prob: float,
        traits: list[float],
        epistemic_uncertainty: float,
        nodule_radius_mm: float,
    ) -> dict[str, Any]:
        values = [calibrated_prob, epistemic_uncertainty, nodule_radius_mm, *traits]
        if not all(isfinite(float(value)) for value in values):
            raise ValueError("Classification inputs must be finite numbers.")
        if not 0.0 <= calibrated_prob <= 1.0:
            raise ValueError("calibrated_prob must be between 0 and 1.")
        if epistemic_uncertainty < 0.0:
            raise ValueError("epistemic_uncertainty cannot be negative.")
        if nodule_radius_mm < 0.0:
            raise ValueError("nodule_radius_mm cannot be negative.")
        if len(traits) != 6:
            raise ValueError("traits must contain six radiological predictions.")

        diameter_mm = float(nodule_radius_mm) * 2.0
        if diameter_mm < 6.0:
            category, name, management = (
                "2",
                "Benign Appearance or Behavior",
                "Continue annual screening with low-dose CT in 12 months.",
            )
        elif diameter_mm < 8.0:
            category, name, management = (
                "3",
                "Probably Benign",
                "Follow-up low-dose CT in 6 months.",
            )
        elif diameter_mm < 15.0:
            category, name, management = (
                "4A",
                "Suspicious",
                "Follow-up low-dose CT in 3 months; consider PET/CT when clinically appropriate.",
            )
        else:
            category, name, management = (
                "4B",
                "Very Suspicious",
                "Diagnostic chest CT and specialist evaluation; consider PET/CT or tissue sampling.",
            )

        if calibrated_prob < 0.10:
            risk_tier = "Low"
        elif calibrated_prob < 0.50:
            risk_tier = "Intermediate"
        else:
            risk_tier = "High"

        return {
            "category": category,
            "name": name,
            "risk_tier": risk_tier,
            "nodule_diameter_mm": diameter_mm,
            "management": management,
            "model_malignancy_probability": float(calibrated_prob),
            "epistemic_uncertainty": float(epistemic_uncertainty),
            "radiological_traits": [float(value) for value in traits],
            "assessment_note": (
                "Estimated from diameter alone, assuming a baseline solid nodule. "
                "Nodule type, growth, and prior imaging were not available."
            ),
        }


class RadiologicalReportGenerator:
    """Build an in-memory PDF report for the dashboard's decision-support output."""

    def generate_pdf_bytes(
        self,
        patient_info: dict[str, Any],
        prediction_results: dict[str, Any],
        verification_results: dict[str, Any],
        lung_rads_info: dict[str, Any],
        gradcam_img_bytes: bytes,
    ) -> bytes:
        buffer = BytesIO()
        document = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            rightMargin=0.65 * inch,
            leftMargin=0.65 * inch,
            topMargin=0.6 * inch,
            bottomMargin=0.6 * inch,
            title="Lung CT Decision-Support Report",
        )
        styles = getSampleStyleSheet()
        styles.add(
            ParagraphStyle(
                name="ReportTitle",
                parent=styles["Title"],
                textColor=colors.HexColor("#1E3A8A"),
                alignment=TA_CENTER,
                spaceAfter=12,
            )
        )
        styles.add(
            ParagraphStyle(
                name="SectionHeading",
                parent=styles["Heading2"],
                textColor=colors.HexColor("#2563EB"),
                spaceBefore=12,
                spaceAfter=6,
            )
        )

        story: list[Any] = [
            Paragraph("Lung CT Decision-Support Report", styles["ReportTitle"]),
            Paragraph(
                "For clinical review only. This AI-generated output is not a diagnosis "
                "or a substitute for interpretation by a qualified clinician.",
                styles["BodyText"],
            ),
            Spacer(1, 10),
        ]

        def add_table(rows: list[list[str]]) -> None:
            wrapped_rows = [
                [Paragraph(value, styles["BodyText"]) for value in row]
                for row in rows
            ]
            table = Table(
                wrapped_rows,
                colWidths=[2.0 * inch, 4.8 * inch],
                hAlign="LEFT",
            )
            table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#EFF6FF")),
                        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CBD5E1")),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("LEFTPADDING", (0, 0), (-1, -1), 7),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                        ("TOPPADDING", (0, 0), (-1, -1), 5),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                    ]
                )
            )
            story.append(table)

        story.append(Paragraph("Scan Information", styles["SectionHeading"]))
        add_table(
            [
                ["Patient ID", escape(str(patient_info.get("patient_id", "N/A")))],
                ["Scan date", escape(str(patient_info.get("scan_date", "N/A")))],
                ["Scan format", escape(str(patient_info.get("scan_format", "N/A")))],
                ["Slice count", escape(str(patient_info.get("slice_count", "N/A")))],
            ]
        )

        probability = float(prediction_results["calibrated_prob"])
        story.extend(
            [
                Paragraph("Model Output and Lung-RADS Estimate", styles["SectionHeading"]),
            ]
        )
        add_table(
            [
                ["Estimated category", escape(str(lung_rads_info["category"]))],
                ["Category description", escape(str(lung_rads_info["name"]))],
                ["Model risk tier", escape(str(lung_rads_info["risk_tier"]))],
                ["Malignancy probability", f"{probability:.1%}"],
                [
                    "Estimated diameter",
                    f"{float(lung_rads_info['nodule_diameter_mm']):.1f} mm",
                ],
                ["Suggested follow-up", escape(str(lung_rads_info["management"]))],
                [
                    "Assessment limitation",
                    escape(str(lung_rads_info["assessment_note"])),
                ],
            ]
        )

        story.append(Paragraph("Self-Verification", styles["SectionHeading"]))
        add_table(
            [
                ["Status", escape(str(verification_results.get("status", "N/A")))],
                [
                    "Summary",
                    escape(str(verification_results.get("final_diagnosis", "N/A"))),
                ],
                [
                    "Recommended action",
                    escape(str(verification_results.get("recommended_action", "N/A"))),
                ],
                [
                    "Epistemic uncertainty",
                    f"{float(prediction_results['epistemic_uncertainty']):.6f}",
                ],
            ]
        )
        conflicts = verification_results.get("conflicts", [])
        if conflicts:
            story.append(Spacer(1, 6))
            story.append(
                Paragraph(
                    "<b>Review flags:</b> "
                    + escape("; ".join(str(item) for item in conflicts)),
                    styles["BodyText"],
                )
            )

        if gradcam_img_bytes:
            story.append(Paragraph("Grad-CAM Visualization", styles["SectionHeading"]))
            image_buffer = BytesIO(gradcam_img_bytes)
            image_reader = ImageReader(image_buffer)
            image_width, image_height = image_reader.getSize()
            scale = min(4.0 * inch / image_width, 4.0 * inch / image_height)
            image = Image(
                image_buffer,
                width=image_width * scale,
                height=image_height * scale,
            )
            image.hAlign = "CENTER"
            story.append(image)

        document.build(story)
        return buffer.getvalue()
