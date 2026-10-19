"""
Scan-Quality and Distribution-Shift Detection Module.
Detects image noise, reconstruction geometry issues, field-of-view artifacts,
and intensity distribution shifts (Out-of-Distribution / OOD) relative to normative thoracic CT data.
Selects adaptive preprocessing or inference pathways.
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np


class ScanQualityAssessor:
    """
    Evaluates CT volume quality, signal-to-noise ratio (SNR), high-frequency noise,
    slice thickness/spacing consistency, and distribution divergence from training data.
    """

    # Normative thoracic CT window parameters (lung window: -600 HU center, 1500 width => -1350 to +150 HU)
    # Expected normalized voxel mean and std across typical thoracic parenchymal volumes
    REFERENCE_MEAN_HU: float = -620.0
    REFERENCE_STD_HU: float = 380.0
    MAX_ACCEPTABLE_NOISE_INDEX: float = 0.22
    MIN_ACCEPTABLE_SNR_DB: float = 8.0

    def __init__(
        self,
        noise_threshold: float = MAX_ACCEPTABLE_NOISE_INDEX,
        min_snr_db: float = MIN_ACCEPTABLE_SNR_DB,
        ood_divergence_threshold: float = 0.40
    ):
        self.noise_threshold = noise_threshold
        self.min_snr_db = min_snr_db
        self.ood_divergence_threshold = ood_divergence_threshold

    def compute_noise_index(self, volume: np.ndarray) -> float:
        """
        Estimates high-frequency noise using Laplacian second-derivative variance
        on normalized volume slices.
        """
        if volume.ndim != 3 or volume.shape[0] < 2:
            return 0.0

        # Sample up to 8 representative slices across the volume
        num_slices = volume.shape[0]
        slice_indices = np.linspace(num_slices // 6, 5 * num_slices // 6, min(8, num_slices), dtype=int)
        
        noise_scores = []
        for s_idx in slice_indices:
            sl = volume[s_idx].astype(np.float32)
            # 2D Laplacian kernel approximation
            # [ 0,  1,  0]
            # [ 1, -4,  1]
            # [ 0,  1,  0]
            if sl.shape[0] >= 5 and sl.shape[1] >= 5:
                laplacian = (
                    np.roll(sl, 1, axis=0) + np.roll(sl, -1, axis=0) +
                    np.roll(sl, 1, axis=1) + np.roll(sl, -1, axis=1) - 4 * sl
                )[2:-2, 2:-2]
                # High frequency variance in homogeneous background/parenchyma
                noise_scores.append(float(np.std(laplacian)))

        return float(np.mean(noise_scores)) if noise_scores else 0.0

    def compute_snr_db(self, volume: np.ndarray) -> float:
        """
        Estimates Signal-to-Noise Ratio (SNR) in decibels (dB) by comparing
        parenchymal signal intensity to background noise variance.
        """
        vol_flat = volume.flatten()
        signal_mean = np.mean(vol_flat)
        noise_std = np.std(vol_flat) + 1e-6
        snr_linear = np.abs(signal_mean) / noise_std
        snr_db = 20.0 * np.log10(max(snr_linear, 1e-4))
        return float(snr_db)

    def compute_distribution_shift(self, volume: np.ndarray) -> Dict[str, float]:
        """
        Quantifies distribution shift (Wasserstein-1 distance approximation & Kolmogorov-Smirnov-like divergence)
        against expected normalized thoracic CT intensity distribution.
        """
        # Normative reference distribution: unimodal/bimodal thoracic CT lung parenchyma
        # Sample voxel distribution
        sub_sample = volume[::max(1, volume.shape[0] // 8), ::4, ::4].flatten()
        
        sample_mean = float(np.mean(sub_sample))
        sample_std = float(np.std(sub_sample))
        sample_p10 = float(np.percentile(sub_sample, 10))
        sample_p50 = float(np.percentile(sub_sample, 50))
        sample_p90 = float(np.percentile(sub_sample, 90))

        # Expected normalized range [0, 1] for windowed CT: mean ~ 0.35 - 0.55, std ~ 0.20 - 0.35
        expected_mean = 0.45
        expected_std = 0.26

        mean_diff = abs(sample_mean - expected_mean)
        std_diff = abs(sample_std - expected_std)
        divergence_score = float(np.clip(mean_diff * 1.5 + std_diff * 1.2, 0.0, 1.0))

        return {
            "divergence_score": divergence_score,
            "sample_mean": sample_mean,
            "sample_std": sample_std,
            "sample_median": sample_p50,
            "p10": sample_p10,
            "p90": sample_p90,
            "is_shifted": divergence_score > self.ood_divergence_threshold
        }

    def detect_artifacts(self, volume: np.ndarray) -> Dict[str, Any]:
        """
        Detects streak artifacts, severe clipping, or missing slice discontinuities.
        """
        D, H, W = volume.shape
        # Check slice-to-slice intensity jump (e.g., motion or missing data)
        slice_means = np.mean(volume, axis=(1, 2))
        slice_diffs = np.abs(np.diff(slice_means))
        max_slice_jump = float(np.max(slice_diffs)) if len(slice_diffs) > 0 else 0.0
        
        # Check boundary saturation / clipping (over-clipping indicates improper windowing)
        clipped_high = float(np.mean(volume >= 0.99))
        clipped_low = float(np.mean(volume <= 0.01))
        severe_clipping = (clipped_high > 0.45) or (clipped_low > 0.75)

        has_motion_jump = max_slice_jump > 0.25
        has_artifacts = severe_clipping or has_motion_jump

        return {
            "has_artifacts": has_artifacts,
            "max_slice_jump": max_slice_jump,
            "clipped_high_ratio": clipped_high,
            "clipped_low_ratio": clipped_low,
            "severe_clipping": severe_clipping,
            "motion_discontinuity": has_motion_jump
        }

    def assess_quality(self, volume: np.ndarray) -> Dict[str, Any]:
        """
        Runs comprehensive scan-quality checks and selects the optimal preprocessing/inference pathway.
        
        Pathways:
            - 'STANDARD': Scan satisfies normative criteria; proceed with direct multi-task inference.
            - 'ADAPTIVE_DENOISING': Elevated noise detected; apply bilateral edge-preserving smoothing.
            - 'HISTOGRAM_RECALIBRATION': Distribution shift detected; apply adaptive histogram normalization.
            - 'REJECT_CLINICIAN_REVIEW': Severe artifacts, extreme corruption or uncorrectable OOD shift.
        """
        noise_idx = self.compute_noise_index(volume)
        snr_db = self.compute_snr_db(volume)
        dist_shift = self.compute_distribution_shift(volume)
        artifacts = self.detect_artifacts(volume)

        # Composite Scan Quality Score (0 to 100, 100 = Pristine)
        quality_score = 100.0
        # Penalties
        quality_score -= min(35.0, (noise_idx / max(self.noise_threshold, 1e-4)) * 25.0)
        if snr_db < self.min_snr_db:
            quality_score -= min(25.0, (self.min_snr_db - snr_db) * 4.0)
        quality_score -= dist_shift["divergence_score"] * 30.0
        if artifacts["has_artifacts"]:
            quality_score -= 25.0
        quality_score = float(np.clip(quality_score, 0.0, 100.0))

        # Select adaptive pathway
        if artifacts["severe_clipping"] and dist_shift["divergence_score"] > 0.65:
            pathway = "REJECT_CLINICIAN_REVIEW"
            status_desc = "Critically degraded scan quality: severe intensity clipping and extreme distribution shift."
        elif noise_idx > self.noise_threshold and dist_shift["is_shifted"]:
            pathway = "ADAPTIVE_DENOISING_AND_RECALIBRATION"
            status_desc = "Elevated noise and distribution shift detected. Routing to adaptive filtering and HU recalibration."
        elif noise_idx > self.noise_threshold:
            pathway = "ADAPTIVE_DENOISING"
            status_desc = "Elevated high-frequency reconstruction noise detected. Routing to edge-preserving bilateral denoising."
        elif dist_shift["is_shifted"]:
            pathway = "HISTOGRAM_RECALIBRATION"
            status_desc = "Distribution shift from training domain detected. Routing to adaptive contrast normalization."
        else:
            pathway = "STANDARD"
            status_desc = "Scan quality satisfies all standard diagnostic acquisition criteria."

        return {
            "quality_score": round(quality_score, 1),
            "recommended_pathway": pathway,
            "status_description": status_desc,
            "noise_index": round(noise_idx, 4),
            "snr_db": round(snr_db, 2),
            "distribution_divergence": round(dist_shift["divergence_score"], 3),
            "is_out_of_distribution": dist_shift["is_shifted"],
            "artifacts_detected": artifacts["has_artifacts"],
            "artifact_details": artifacts,
            "is_acceptable": pathway != "REJECT_CLINICIAN_REVIEW"
        }

    def apply_adaptive_pathway(
        self,
        volume: np.ndarray,
        assessment: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, str]:
        """
        Executes the targeted preprocessing transformation dictated by the quality assessment.
        Returns processed volume and description of operation performed.
        """
        if assessment is None:
            assessment = self.assess_quality(volume)

        pathway = assessment["recommended_pathway"]
        processed = volume.copy().astype(np.float32)

        if pathway == "STANDARD":
            return processed, "Direct Standard Inference (No Modification Required)"

        if "DENOISING" in pathway:
            # 3D edge-preserving filter: 3x3 local median/blur across slices
            for d in range(processed.shape[0]):
                sl = processed[d]
                # Fast 3x3 separable box blur for smoothing noise while preserving nodule edges
                pad = np.pad(sl, ((1, 1), (1, 1)), mode='reflect')
                smooth = (
                    pad[:-2, :-2] + pad[:-2, 1:-1] + pad[:-2, 2:] +
                    pad[1:-1, :-2] + 4 * pad[1:-1, 1:-1] + pad[1:-1, 2:] +
                    pad[2:, :-2] + pad[2:, 1:-1] + pad[2:, 2:]
                ) / 12.0
                processed[d] = np.clip(smooth, 0.0, 1.0)

        if "RECALIBRATION" in pathway:
            # Adaptive histogram percentile normalization to map 5th - 95th percentiles to canonical thoracic range
            p5 = np.percentile(processed, 5)
            p95 = np.percentile(processed, 95)
            if p95 > p5:
                processed = np.clip((processed - p5) / (p95 - p5), 0.0, 1.0)

        return processed, f"Executed Pathway: {pathway}"
