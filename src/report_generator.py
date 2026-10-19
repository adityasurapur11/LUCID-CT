"""Lung-RADS decision-support classification and clinical PDF report generation."""

from __future__ import annotations

from io import BytesIO
from math import isfinite
from pathlib import Path
from typing import Any

from PIL import Image as PILImage
from PIL import ImageDraw, ImageFont
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader


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
    """Build matching, single-page PNG and PDF decision-support reports."""

    _PAGE_SIZE = (1500, 1938)
    _COLORS = {
        "background": "#F3F6FA",
        "white": "#FFFFFF",
        "navy": "#102A43",
        "blue": "#176B87",
        "teal": "#13A89E",
        "text": "#203447",
        "muted": "#62778A",
        "line": "#D9E2EC",
        "soft_blue": "#EAF3F8",
        "soft_teal": "#E8F6F3",
        "soft_amber": "#FFF5E5",
        "amber": "#A9610B",
        "soft_red": "#FCEBEC",
        "red": "#A83942",
    }

    @staticmethod
    def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
        candidates = (
            (
                r"C:\Windows\Fonts\arialbd.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
            )
            if bold
            else (
                r"C:\Windows\Fonts\arial.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
            )
        )
        for candidate in candidates:
            if Path(candidate).is_file():
                return ImageFont.truetype(candidate, size=size)
        return ImageFont.load_default()

    @staticmethod
    def _wrap_text(
        draw: ImageDraw.ImageDraw,
        text: str,
        font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
        max_width: int,
    ) -> list[str]:
        words = str(text).split()
        if not words:
            return [""]

        lines: list[str] = []
        current = ""
        for word in words:
            candidate = f"{current} {word}".strip()
            if draw.textbbox((0, 0), candidate, font=font)[2] <= max_width:
                current = candidate
                continue
            if current:
                lines.append(current)
            current = ""
            for character in word:
                candidate = current + character
                if draw.textbbox((0, 0), candidate, font=font)[2] > max_width and current:
                    lines.append(current)
                    current = character
                else:
                    current = candidate
        if current:
            lines.append(current)
        return lines

    def _render_report(
        self,
        patient_info: dict[str, Any],
        prediction_results: dict[str, Any],
        verification_results: dict[str, Any],
        lung_rads_info: dict[str, Any],
        gradcam_img_bytes: bytes,
        adaptive_results: dict[str, Any] | None = None,
    ) -> bytes:
        width, height = self._PAGE_SIZE
        image = PILImage.new("RGB", (width, height), self._COLORS["background"])
        draw = ImageDraw.Draw(image)
        colors = self._COLORS
        margin = 72
        gap = 28
        column_width = (width - 2 * margin - gap) // 2

        def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
            return self._font(size, bold)

        def rounded_box(
            box: tuple[int, int, int, int],
            fill: str = "white",
            outline: str | None = None,
            radius: int = 22,
        ) -> None:
            draw.rounded_rectangle(
                box,
                radius=radius,
                fill=colors[fill],
                outline=colors[outline] if outline else None,
                width=2 if outline else 1,
            )

        def wrapped(
            text: str,
            x: int,
            y: int,
            max_width: int,
            size: int = 20,
            color: str = "text",
            bold: bool = False,
            line_gap: int = 7,
        ) -> int:
            selected_font = font(size, bold)
            lines = self._wrap_text(draw, text, selected_font, max_width)
            line_height = size + line_gap
            for line in lines:
                draw.text((x, y), line, font=selected_font, fill=colors[color])
                y += line_height
            return y

        def section_title(text: str, x: int, y: int) -> int:
            draw.text((x, y), text.upper(), font=font(20, True), fill=colors["blue"])
            return y + 42

        # Header
        draw.rectangle((0, 0, width, 202), fill=colors["navy"])
        draw.rectangle((0, 194, width, 202), fill=colors["teal"])
        draw.text((margin, 38), "LUNG CT  /  ADAPTIVE VERIFIED DECISION SUPPORT", font=font(17, True), fill="#9FE3DB")
        draw.text((margin, 76), "Clinical Imaging & Verification Report", font=font(43, True), fill=colors["white"])
        draw.text(
            (margin, 137),
            "Uncertainty-Calibrated & Adaptive Self-Verifying Pulmonary Assessment",
            font=font(20),
            fill="#D9E7F2",
        )
        draw.text(
            (width - margin - 245, 48),
            "REPORT",
            font=font(16, True),
            fill="#9FE3DB",
        )
        raw_date = str(patient_info.get("scan_date", "N/A"))
        clean_date = f"{raw_date[:4]}-{raw_date[4:6]}-{raw_date[6:]}" if len(raw_date) == 8 and raw_date.isdigit() else raw_date
        draw.text(
            (width - margin - 245, 77),
            clean_date,
            font=font(23, True),
            fill=colors["white"],
        )

        # Patient and scan metadata
        rounded_box((margin, 228, width - margin, 364), outline="line")
        draw.text((margin + 26, 248), "PATIENT & SCAN PROFILE", font=font(17, True), fill=colors["blue"])
        metadata = (
            ("Patient ID", str(patient_info.get("patient_id", "N/A"))),
            ("Scan date", clean_date),
            ("Scan format", str(patient_info.get("scan_format", "N/A"))),
            ("Slices", str(patient_info.get("slice_count", "N/A"))),
        )
        cell_width = (width - 2 * margin - 52) // len(metadata)
        for index, (label, value) in enumerate(metadata):
            x = margin + 26 + index * cell_width
            draw.text((x, 284), label.upper(), font=font(13, True), fill=colors["muted"])
            wrapped(value, x, 309, cell_width - 16, size=17, bold=True, line_gap=3)

        # Main result summary
        summary_top = 388
        summary_bottom = 580
        rounded_box((margin, summary_top, width - margin, summary_bottom), fill="soft_blue")
        draw.rounded_rectangle(
            (margin, summary_top, margin + 12, summary_bottom),
            radius=6,
            fill=colors["teal"],
        )
        draw.text(
            (margin + 36, summary_top + 23),
            "ESTIMATED CATEGORY",
            font=font(15, True),
            fill=colors["blue"],
        )
        category = str(lung_rads_info["category"])
        draw.text((margin + 36, summary_top + 49), category, font=font(64, True), fill=colors["navy"])
        draw.text(
            (margin + 145, summary_top + 69),
            str(lung_rads_info["name"]),
            font=font(24, True),
            fill=colors["navy"],
        )
        probability = float(prediction_results["calibrated_prob"])
        draw.text(
            (margin + 36, summary_top + 133),
            f"Calibrated Malignancy Risk: {probability:.1%}",
            font=font(19, True),
            fill=colors["text"],
        )
        draw.text(
            (width - margin - 310, summary_top + 34),
            "RISK TIER",
            font=font(15, True),
            fill=colors["muted"],
        )
        tier_color = "red" if lung_rads_info["risk_tier"] == "High" else "amber"
        tier_fill = "soft_red" if tier_color == "red" else "soft_amber"
        draw.rounded_rectangle(
            (width - margin - 310, summary_top + 61, width - margin - 38, summary_top + 113),
            radius=18,
            fill=colors[tier_fill],
        )
        draw.text(
            (width - margin - 290, summary_top + 73),
            str(lung_rads_info["risk_tier"]).upper(),
            font=font(23, True),
            fill=colors[tier_color],
        )
        diameter = float(lung_rads_info["nodule_diameter_mm"])
        draw.text(
            (width - margin - 310, summary_top + 137),
            f"Estimated diameter: {diameter:.1f} mm",
            font=font(17, True),
            fill=colors["text"],
        )

        left_x = margin
        right_x = margin + column_width + gap
        content_top = 608
        card_bottom = 1560
        rounded_box((left_x, content_top, left_x + column_width, card_bottom), outline="line")
        rounded_box((right_x, content_top, right_x + column_width, card_bottom), outline="line")

        # Clinical verification and actions
        y = section_title("Adaptive Verification & Reliability Gate", left_x + 26, content_top + 25)
        
        # Reliability Gate Status
        if adaptive_results:
            gate_status = adaptive_results.get("final_status", "N/A")
            is_cleared = adaptive_results.get("is_cleared_by_gate", False)
            cri_score = adaptive_results.get("composite_reliability_index", 0.0)
            status_text_display = f"{gate_status} (CRI: {cri_score:.1f}%)"
            status_fill = "soft_teal" if is_cleared else "soft_red"
            status_text_color = "teal" if is_cleared else "red"
        else:
            status_text_display = str(verification_results.get("status", "N/A")).replace("_", " ")
            status_fill = "soft_teal" if verification_results.get("is_verified") else "soft_amber"
            status_text_color = "teal" if verification_results.get("is_verified") else "amber"

        draw.rounded_rectangle(
            (left_x + 26, y, left_x + column_width - 26, y + 44),
            radius=14,
            fill=colors[status_fill],
        )
        draw.text((left_x + 36, y + 10), status_text_display[:38], font=font(15, True), fill=colors[status_text_color])
        y += 56

        diagnosis_text = adaptive_results.get("final_status") if adaptive_results else verification_results.get("final_diagnosis", "Not available")
        y = wrapped(
            str(diagnosis_text),
            left_x + 26,
            y,
            column_width - 52,
            size=20,
            bold=True,
        )
        y += 8
        y = wrapped(
            "Recommended Clinical Action",
            left_x + 26,
            y,
            column_width - 52,
            size=15,
            color="muted",
            bold=True,
        )
        action_text = adaptive_results.get("recommended_clinical_action") if adaptive_results else verification_results.get("recommended_action", "Not available")
        y = wrapped(
            str(action_text),
            left_x + 26,
            y + 2,
            column_width - 52,
            size=16,
        )
        y += 10

        # If re-analysis audit was executed, display reanalysis details
        if adaptive_results and adaptive_results.get("reanalysis_performed"):
            draw.line(
                (left_x + 26, y, left_x + column_width - 26, y),
                fill=colors["line"],
                width=1,
            )
            y += 8
            draw.text((left_x + 26, y), "TARGETED RE-ANALYSIS AUDIT TRAIL", font=font(14, True), fill=colors["blue"])
            y += 24
            agr = adaptive_results.get("reanalysis_agreement", 1.0) * 100.0
            draw.text((left_x + 26, y), f"Agreement with Baseline: {agr:.1f}%", font=font(14, True), fill=colors["navy"])
            y += 22
            re_audit = adaptive_results.get("reanalysis_details", {}).get("reanalysis_audit", [])
            for r_item in re_audit[:2]:
                op_name = r_item.get("target_scale") or r_item.get("target_mode") or r_item.get("operation")
                draw.text((left_x + 26, y), f"* {op_name[:36]}", font=font(13), fill=colors["text"])
                y += 20
            y += 6

        draw.line(
            (left_x + 26, y, left_x + column_width - 26, y),
            fill=colors["line"],
            width=2,
        )
        y = section_title("Uncertainty & Verification Indices", left_x + 26, y + 14)
        uncertainty_rows = [
            ("Epistemic Uncertainty", f"{float(prediction_results['epistemic_uncertainty']):.4f}"),
            ("Aleatoric Uncertainty", f"{float(prediction_results['aleatoric_uncertainty']):.4f}"),
            ("Predictive Entropy", f"{float(prediction_results['total_uncertainty']):.4f}"),
        ]
        if adaptive_results:
            sq = adaptive_results.get("scan_quality", {})
            uncertainty_rows.append(("Scan Quality Score", f"{sq.get('quality_score', 90):.1f} / 100"))
            xf = adaptive_results.get("xai_fidelity", {})
            uncertainty_rows.append(("Grad-CAM Fidelity", f"{xf.get('fidelity_score', 0.8):.2f} / 1.0"))
            tg = adaptive_results.get("trait_grounding", {})
            uncertainty_rows.append(("Trait Physical Grounding", f"{tg.get('composite_trait_grounding', 0.85):.2f} / 1.0"))

        for label, val_str in uncertainty_rows:
            draw.text((left_x + 26, y), label, font=font(15), fill=colors["text"])
            draw.text(
                (left_x + column_width - 150, y),
                val_str,
                font=font(15, True),
                fill=colors["navy"],
            )
            y += 26

        # Radiological characteristics and explainability image
        y = section_title("Radiological Trait Evidence", right_x + 26, content_top + 25)
        trait_breakdown = verification_results.get("trait_breakdown", {})
        trait_rows = (
            ("Spiculation", "Spiculation (1-5)"),
            ("Lobulation", "Lobulation (1-5)"),
            ("Calcification", "Calcification (1-6)"),
            ("Subtlety", "Subtlety (1-5)"),
            ("Margin", "Margin (1-5)"),
            ("Sphericity", "Sphericity (1-5)"),
        )
        for index, (label, key) in enumerate(trait_rows):
            row_top = y + index * 44
            if index % 2 == 0:
                draw.rounded_rectangle(
                    (right_x + 20, row_top - 4, right_x + column_width - 20, row_top + 34),
                    radius=10,
                    fill=colors["background"],
                )
            draw.text((right_x + 32, row_top + 3), label, font=font(16), fill=colors["text"])
            value = str(trait_breakdown.get(key, "N/A"))
            draw.text(
                (right_x + column_width - 150, row_top + 3),
                value,
                font=font(16, True),
                fill=colors["navy"],
            )

        y += len(trait_rows) * 44 + 8
        draw.line(
            (right_x + 26, y, right_x + column_width - 26, y),
            fill=colors["line"],
            width=2,
        )
        y = section_title("XAI Localization & Evidence Grounding", right_x + 26, y + 14)
        cam_box = (right_x + 26, y, right_x + column_width - 26, y + 280)
        draw.rounded_rectangle(cam_box, radius=14, fill=colors["background"])
        if gradcam_img_bytes:
            with PILImage.open(BytesIO(gradcam_img_bytes)) as cam:
                cam = cam.convert("RGB")
                cam.thumbnail((cam_box[2] - cam_box[0] - 20, cam_box[3] - cam_box[1] - 20))
                paste_x = cam_box[0] + (cam_box[2] - cam_box[0] - cam.width) // 2
                paste_y = cam_box[1] + (cam_box[3] - cam_box[1] - cam.height) // 2
                image.paste(cam, (paste_x, paste_y))
        else:
            fallback = "No Grad-CAM image supplied"
            fallback_font = font(16)
            bounds = draw.textbbox((0, 0), fallback, font=fallback_font)
            draw.text(
                (
                    cam_box[0] + (cam_box[2] - cam_box[0] - (bounds[2] - bounds[0])) // 2,
                    cam_box[1] + (cam_box[3] - cam_box[1] - (bounds[3] - bounds[1])) // 2,
                ),
                fallback,
                font=fallback_font,
                fill=colors["muted"],
            )
        y = cam_box[3] + 12
        y = wrapped(
            str(lung_rads_info["assessment_note"]),
            right_x + 26,
            y,
            column_width - 52,
            size=14,
            color="muted",
            line_gap=4,
        )

        # Fixed footer
        footer_y = 1600
        draw.line(
            (margin, footer_y, width - margin, footer_y),
            fill=colors["line"],
            width=2,
        )
        wrapped(
            "Adaptive Self-Verifying Decision Support System. Features targeted computational re-analysis, "
            "radiological trait physical grounding, and selective prediction reliability gating. "
            "For investigational review by certified thoracic medical specialists.",
            margin,
            footer_y + 15,
            width - 2 * margin,
            size=14,
            color="muted",
            line_gap=4,
        )

        output = BytesIO()
        image.save(output, format="PNG", optimize=True)
        return output.getvalue()

    def generate_image_bytes(
        self,
        patient_info: dict[str, Any],
        prediction_results: dict[str, Any],
        verification_results: dict[str, Any],
        lung_rads_info: dict[str, Any],
        gradcam_img_bytes: bytes,
        adaptive_results: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> bytes:
        return self._render_report(
            patient_info,
            prediction_results,
            verification_results,
            lung_rads_info,
            gradcam_img_bytes,
            adaptive_results=adaptive_results,
        )

    def generate_pdf_bytes(
        self,
        patient_info: dict[str, Any],
        prediction_results: dict[str, Any],
        verification_results: dict[str, Any],
        lung_rads_info: dict[str, Any],
        gradcam_img_bytes: bytes,
        adaptive_results: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> bytes:
        report_image = self._render_report(
            patient_info,
            prediction_results,
            verification_results,
            lung_rads_info,
            gradcam_img_bytes,
            adaptive_results=adaptive_results,
        )
        buffer = BytesIO()
        pdf = canvas.Canvas(buffer, pagesize=letter, pageCompression=1)
        pdf.setTitle("Adaptive Self-Verifying Lung CT Decision-Support Report")
        page_width, page_height = letter
        pdf.drawImage(
            ImageReader(BytesIO(report_image)),
            0,
            0,
            width=page_width,
            height=page_height,
            preserveAspectRatio=False,
        )
        pdf.showPage()
        pdf.save()
        return buffer.getvalue()
