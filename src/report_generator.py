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
        draw.text((margin, 38), "LUNG CT  /  DECISION SUPPORT", font=font(17, True), fill="#9FE3DB")
        draw.text((margin, 76), "Clinical Imaging Report", font=font(43, True), fill=colors["white"])
        draw.text(
            (margin, 137),
            "AI-assisted findings for review by a qualified clinician",
            font=font(20),
            fill="#D9E7F2",
        )
        draw.text(
            (width - margin - 245, 48),
            "REPORT",
            font=font(16, True),
            fill="#9FE3DB",
        )
        draw.text(
            (width - margin - 245, 77),
            str(patient_info.get("scan_date", "N/A")),
            font=font(23, True),
            fill=colors["white"],
        )

        # Patient and scan metadata
        rounded_box((margin, 228, width - margin, 364), outline="line")
        draw.text((margin + 26, 248), "PATIENT & SCAN", font=font(17, True), fill=colors["blue"])
        metadata = (
            ("Patient ID", str(patient_info.get("patient_id", "N/A"))),
            ("Scan date", str(patient_info.get("scan_date", "N/A"))),
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
            f"Model malignancy estimate: {probability:.1%}",
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
        y = section_title("Clinical review", left_x + 26, content_top + 25)
        status = str(verification_results.get("status", "N/A")).replace("_", " ")
        status_fill = "soft_teal" if verification_results.get("is_verified") else "soft_amber"
        status_text = "teal" if verification_results.get("is_verified") else "amber"
        draw.rounded_rectangle(
            (left_x + 26, y, left_x + column_width - 26, y + 44),
            radius=14,
            fill=colors[status_fill],
        )
        draw.text((left_x + 42, y + 10), status, font=font(17, True), fill=colors[status_text])
        y += 60
        y = wrapped(
            str(verification_results.get("final_diagnosis", "Not available")),
            left_x + 26,
            y,
            column_width - 52,
            size=22,
            bold=True,
        )
        y += 8
        y = wrapped(
            "Recommended action",
            left_x + 26,
            y,
            column_width - 52,
            size=15,
            color="muted",
            bold=True,
        )
        y = wrapped(
            str(verification_results.get("recommended_action", "Not available")),
            left_x + 26,
            y + 2,
            column_width - 52,
            size=18,
        )
        y += 14
        y = wrapped(
            "Suggested nodule follow-up",
            left_x + 26,
            y,
            column_width - 52,
            size=15,
            color="muted",
            bold=True,
        )
        y = wrapped(
            str(lung_rads_info["management"]),
            left_x + 26,
            y + 2,
            column_width - 52,
            size=18,
        )

        y += 12
        draw.line(
            (left_x + 26, y, left_x + column_width - 26, y),
            fill=colors["line"],
            width=2,
        )
        y = section_title("Uncertainty", left_x + 26, y + 16)
        uncertainty_rows = (
            ("Epistemic (model)", prediction_results["epistemic_uncertainty"]),
            ("Aleatoric (data)", prediction_results["aleatoric_uncertainty"]),
            ("Total predictive entropy", prediction_results["total_uncertainty"]),
        )
        for label, value in uncertainty_rows:
            draw.text((left_x + 26, y), label, font=font(16), fill=colors["text"])
            draw.text(
                (left_x + column_width - 150, y),
                f"{float(value):.4f}",
                font=font(16, True),
                fill=colors["navy"],
            )
            y += 32

        conflicts = verification_results.get("conflicts", [])
        y += 10
        draw.line(
            (left_x + 26, y, left_x + column_width - 26, y),
            fill=colors["line"],
            width=2,
        )
        y = section_title("Review flags", left_x + 26, y + 16)
        if conflicts:
            for conflict in conflicts:
                y = wrapped(
                    f"- {conflict}",
                    left_x + 26,
                    y,
                    column_width - 52,
                    size=15,
                    color="red",
                    line_gap=5,
                )
                y += 5
        else:
            wrapped(
                "No additional verification flags.",
                left_x + 26,
                y,
                column_width - 52,
                size=16,
                color="muted",
            )

        # Radiological characteristics and explainability image
        y = section_title("Radiological traits", right_x + 26, content_top + 25)
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
            row_top = y + index * 48
            if index % 2 == 0:
                draw.rounded_rectangle(
                    (right_x + 20, row_top - 4, right_x + column_width - 20, row_top + 36),
                    radius=10,
                    fill=colors["background"],
                )
            draw.text((right_x + 32, row_top + 3), label, font=font(17), fill=colors["text"])
            value = str(trait_breakdown.get(key, "N/A"))
            draw.text(
                (right_x + column_width - 160, row_top + 3),
                value,
                font=font(17, True),
                fill=colors["navy"],
            )

        y += len(trait_rows) * 48 + 10
        draw.line(
            (right_x + 26, y, right_x + column_width - 26, y),
            fill=colors["line"],
            width=2,
        )
        y = section_title("Grad-CAM visualization", right_x + 26, y + 16)
        cam_box = (right_x + 26, y, right_x + column_width - 26, y + 300)
        draw.rounded_rectangle(cam_box, radius=14, fill=colors["background"])
        if gradcam_img_bytes:
            with PILImage.open(BytesIO(gradcam_img_bytes)) as cam:
                cam = cam.convert("RGB")
                cam.thumbnail((cam_box[2] - cam_box[0] - 24, cam_box[3] - cam_box[1] - 24))
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
        y = cam_box[3] + 15
        y = wrapped(
            str(lung_rads_info["assessment_note"]),
            right_x + 26,
            y,
            column_width - 52,
            size=15,
            color="muted",
            line_gap=5,
        )

        # Fixed footer remains outside the content cards and appears on every report.
        footer_y = 1600
        draw.line(
            (margin, footer_y, width - margin, footer_y),
            fill=colors["line"],
            width=2,
        )
        wrapped(
            "Decision-support estimate only — not a diagnosis or official Lung-RADS "
            "assessment. Category assumes a baseline solid nodule; nodule type, growth, "
            "and prior imaging were not available. Review the original scan and consult "
            "a qualified clinician.",
            margin,
            footer_y + 17,
            width - 2 * margin,
            size=15,
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
    ) -> bytes:
        return self._render_report(
            patient_info,
            prediction_results,
            verification_results,
            lung_rads_info,
            gradcam_img_bytes,
        )

    def generate_pdf_bytes(
        self,
        patient_info: dict[str, Any],
        prediction_results: dict[str, Any],
        verification_results: dict[str, Any],
        lung_rads_info: dict[str, Any],
        gradcam_img_bytes: bytes,
    ) -> bytes:
        report_image = self._render_report(
            patient_info,
            prediction_results,
            verification_results,
            lung_rads_info,
            gradcam_img_bytes,
        )
        buffer = BytesIO()
        pdf = canvas.Canvas(buffer, pagesize=letter, pageCompression=1)
        pdf.setTitle("Lung CT Decision-Support Report")
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
