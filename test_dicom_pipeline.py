import os
import glob
import time
import numpy as np
import pydicom
import torch

from src.preprocessing import apply_hu_window, segment_lung_mask_slice
from src.model import HybridLungCancerModel
from src.uncertainty_calibration import estimate_mc_uncertainty, TemperatureScaler
from src.xai import GradCAM3D, overlay_heatmap_on_slice
from src.self_verification import SelfVerificationEngine
from src.scan_quality import ScanQualityAssessor
from src.trait_verifier import TraitSpecificRegionalVerifier
from src.xai_verifier import ExplanationFidelityVerifier
from src.adaptive_controller import AdaptiveVerificationController
from src.report_generator import LungRADSClassifier, RadiologicalReportGenerator

# 1. Load sample DICOM
workspace_dir = os.path.dirname(os.path.abspath(__file__))
test_file = os.path.join(workspace_dir, "1.dcm")
if not os.path.exists(test_file):
    candidates = glob.glob(os.path.join(workspace_dir, "**", "*.dcm"), recursive=True)
    if candidates:
        test_file = candidates[0]
    else:
        raise FileNotFoundError("No .dcm files found for testing.")

print(f"Testing with DICOM file: {test_file}")

ds = pydicom.dcmread(test_file)
raw_slice = ds.pixel_array.astype(np.float32)
slope = float(getattr(ds, "RescaleSlope", 1))
intercept = float(getattr(ds, "RescaleIntercept", -1024))
hu_slice = raw_slice * slope + intercept
windowed_slice = apply_hu_window(hu_slice)

print(f"Patient ID: {getattr(ds, 'PatientID', 'N/A')}, Study Date: {getattr(ds, 'StudyDate', 'N/A')}")
print(f"Raw slice shape: {raw_slice.shape}, HU min/max: {hu_slice.min():.1f} / {hu_slice.max():.1f}")
print(f"Windowed slice min/max: {windowed_slice.min():.2f} / {windowed_slice.max():.2f}")

# Stack single slice to 3D volume (16, H, W)
vol = np.tile(windowed_slice, (16, 1, 1))

# Check size
D, H, W = vol.shape
print(f"Volume shape: {vol.shape}")

# Model and scaler
model = HybridLungCancerModel(in_channels=1)
ckpt_path = os.path.join(workspace_dir, "models", "hybrid_lung_cancer_model.pt")
temp_val = 1.2
if os.path.exists(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
        temp_val = float(ckpt.get("temperature", 1.2))
        print("Loaded trained model checkpoint (hybrid_lung_cancer_model.pt).")
    elif isinstance(ckpt, dict):
        model.load_state_dict(ckpt)
model.eval()
scaler = TemperatureScaler(temperature=temp_val)

# Test 1: Scan Quality
quality_assessor = ScanQualityAssessor()
quality_res = quality_assessor.assess_quality(vol)
print("\n--- SCAN QUALITY ---")
print("Quality Score:", quality_res["quality_score"])
print("Pathway:", quality_res["recommended_pathway"])
print("SNR (dB):", quality_res["snr_db"])
print("Noise Index:", quality_res["noise_index"])
print("Distribution Divergence:", quality_res["distribution_divergence"])

# Test 2: Inference & Uncertainty
vol_tensor = torch.tensor(vol, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
t0 = time.time()
unc_results = estimate_mc_uncertainty(model, vol_tensor, num_samples=10, temp_scaler=scaler)
t_inf = time.time() - t0
print(f"\n--- INFERENCE (took {t_inf:.2f}s) ---")
print("Calibrated Malignancy Probability:", f"{unc_results['calibrated_prob']*100:.2f}%")
print("Epistemic Uncertainty:", unc_results["epistemic_uncertainty"])
print("Aleatoric Uncertainty:", unc_results["aleatoric_uncertainty"])
print("Total Uncertainty:", unc_results["total_uncertainty"])
print("Predicted Traits:", [round(t, 2) for t in unc_results["trait_predictions"]])

# Test 3: Trait Grounding
trait_verifier = TraitSpecificRegionalVerifier()
trait_res = trait_verifier.verify_all_traits(vol, unc_results["trait_predictions"])
print("\n--- TRAIT GROUNDING ---")
print("Composite Trait Grounding:", trait_res["composite_trait_grounding"])
print("Individual Scores:", trait_res["individual_grounding_scores"])
print("Conflicts:", trait_res["trait_conflicts"])

# Test 4: Grad-CAM
cam_gen = GradCAM3D(model, model.block2.conv)
heatmap_3d = cam_gen.generate_heatmap(vol_tensor, target_class=1)
print(f"\n--- GRAD-CAM ---")
print("Heatmap shape:", heatmap_3d.shape, "min/max:", heatmap_3d.min(), heatmap_3d.max())

# Test 5: XAI Fidelity
xai_verifier = ExplanationFidelityVerifier()
xai_res = xai_verifier.verify_explanation(model, vol_tensor, heatmap_3d, unc_results["calibrated_prob"], cam_gen)
print("\n--- XAI FIDELITY ---")
print("Fidelity Score (EFI):", xai_res["fidelity_score"])
print("Deletion Passed?:", xai_res["deletion_test"]["passed_deletion"])
print("Attribution Stability:", xai_res["stability_test"]["attribution_stability"])

# Test 6: Adaptive Controller
controller = AdaptiveVerificationController()
adapt_res = controller.run_adaptive_verification_loop(
    model=model,
    volume=vol,
    initial_prob=unc_results["calibrated_prob"],
    epistemic_unc=unc_results["epistemic_uncertainty"],
    initial_traits=unc_results["trait_predictions"],
    heatmap_3d=heatmap_3d,
    cam_generator=cam_gen,
    temp_scaler=scaler
)
print("\n--- ADAPTIVE CONTROLLER ---")
print("Final Status:", adapt_res["final_status"])
print("Cleared by Gate?:", adapt_res["is_cleared_by_gate"])
print("CRI (%):", adapt_res["composite_reliability_index"])
print("Reanalysis Performed?:", adapt_res["reanalysis_performed"])
print("Reanalysis Agreement:", adapt_res["reanalysis_agreement"])

# Test 7: Lung-RADS & Report
lung_rads_info = LungRADSClassifier.classify(
    calibrated_prob=adapt_res["final_calibrated_prob"],
    traits=adapt_res["final_traits"],
    epistemic_uncertainty=unc_results["epistemic_uncertainty"],
    nodule_radius_mm=5.0
)
print("\n--- LUNG-RADS ---")
print("Category:", lung_rads_info["category"], "-", lung_rads_info["name"])
print("Risk Tier:", lung_rads_info["risk_tier"])

report_gen = RadiologicalReportGenerator()
patient_info = {
    "patient_id": getattr(ds, "PatientID", "DICOM-PATIENT-A"),
    "scan_date": "2026-10-19",
    "scan_format": f"Real DICOM Slice ({test_file})",
    "slice_count": D
}
fig_ov = overlay_heatmap_on_slice(vol[D//2], heatmap_3d[D//2])
import cv2
_, img_enc = cv2.imencode('.png', cv2.cvtColor(fig_ov, cv2.COLOR_RGB2BGR))
cam_bytes = img_enc.tobytes()

pdf_bytes = report_gen.generate_pdf_bytes(
    patient_info=patient_info,
    prediction_results=unc_results,
    verification_results={"status": adapt_res["final_status"], "final_diagnosis": adapt_res["final_status"], "recommended_action": adapt_res["recommended_clinical_action"], "trait_breakdown": {}},
    lung_rads_info=lung_rads_info,
    gradcam_img_bytes=cam_bytes,
    adaptive_results=adapt_res
)
print(f"\n--- REPORT ---")
print(f"Generated PDF bytes: {len(pdf_bytes)} bytes")
with open(os.path.join(workspace_dir, "scratch_test_report.pdf"), "wb") as f:
    f.write(pdf_bytes)
print("Saved scratch_test_report.pdf successfully!")
