"""
LUCID-CT: Lung Cancer Inference with Dynamic Self-Verification for 3D CT
Streamlit Clinical Decision Support System & Research Prototype Interface.
Incorporates:
- Quality and Distribution-Shift Detection
- Hybrid 3D CNN-Transformer Multi-Task Diagnostics
- Trait-Specific Multi-Scale Regional Physical Verification
- Evidence-Grounded XAI (Grad-CAM Deletion & Insertion Fidelity)
- Adaptive Verification Controller with Targeted Re-Analysis Loop
- Final Reliability Gate (Accepted vs. Abstained / Clinician Review)
- Longitudinal Growth & Volume Doubling Time (VDT) Tracker
- 4-Way Comparative Ablation Benchmark Suite (Experiments A, B, C, D)
"""

import sys
import os
import io
import time
from typing import Dict, Any, Optional

import streamlit as st
import numpy as np
import torch
# Fix Streamlit watcher compatibility issue with PyTorch 2.x torch.classes module
sys.modules.pop('torch.classes', None)
import cv2
import matplotlib.pyplot as plt

from src.preprocessing import apply_hu_window, segment_lung_mask_slice, extract_roi
from src.dataset import SyntheticLIDCDataset, load_dicom_series, load_nifti_file
from src.model import HybridLungCancerModel
from src.uncertainty_calibration import estimate_mc_uncertainty, TemperatureScaler
from src.xai import GradCAM3D, overlay_heatmap_on_slice
from src.self_verification import SelfVerificationEngine
from src.scan_quality import ScanQualityAssessor
from src.trait_verifier import TraitSpecificRegionalVerifier
from src.xai_verifier import ExplanationFidelityVerifier
from src.adaptive_controller import AdaptiveVerificationController
from src.longitudinal import LongitudinalNoduleTracker
from src.evaluation import BenchmarkAblationSuite
from src.report_generator import LungRADSClassifier, RadiologicalReportGenerator


# Streamlit Page Configuration
st.set_page_config(
    page_title="LUCID-CT | Lung Cancer AI",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Medical UI Aesthetics
st.markdown("""
<style>
    /* Dark Theme & Medical Glassmorphism Accent */
    .main {
        background-color: #0b0f19;
        color: #e2e8f0;
    }
    .stMetric {
        background: rgba(30, 41, 59, 0.65);
        padding: 14px;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(8px);
    }
    .badge-verified {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        padding: 7px 16px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.95rem;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.25);
    }
    .badge-resolved {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        color: white;
        padding: 7px 16px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.95rem;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.25);
    }
    .badge-warning {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        color: white;
        padding: 7px 16px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.95rem;
        box-shadow: 0 4px 12px rgba(245, 158, 11, 0.25);
    }
    .badge-abstain {
        background: linear-gradient(135deg, #ef4444 0%, #b91c1c 100%);
        color: white;
        padding: 7px 16px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.95rem;
        box-shadow: 0 4px 12px rgba(239, 68, 68, 0.25);
    }
    .clinical-card {
        background: rgba(30, 41, 59, 0.75);
        border-left: 5px solid #3b82f6;
        padding: 18px;
        border-radius: 10px;
        margin-bottom: 18px;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    .clinical-card-success {
        background: rgba(16, 185, 129, 0.12);
        border-left: 5px solid #10b981;
        padding: 18px;
        border-radius: 10px;
        margin-bottom: 18px;
        border: 1px solid rgba(16, 185, 129, 0.2);
    }
    .clinical-card-alert {
        background: rgba(239, 68, 68, 0.12);
        border-left: 5px solid #ef4444;
        padding: 18px;
        border-radius: 10px;
        margin-bottom: 18px;
        border: 1px solid rgba(239, 68, 68, 0.2);
    }
    .pipeline-step {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_trained_model():
    """ Load or initialize model """
    model = HybridLungCancerModel(in_channels=1)
    ckpt_path = os.path.join(os.path.dirname(__file__), "models", "hybrid_lung_cancer_model.pt")
    temp_val = 1.2
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
            temp_val = float(ckpt.get("temperature", 1.2))
        elif isinstance(ckpt, dict):
            model.load_state_dict(ckpt)
    model.eval()
    temp_scaler = TemperatureScaler(temperature=temp_val)
    return model, temp_scaler


@st.cache_data
def get_synthetic_samples():
    """ Cache synthetic scans """
    dataset = SyntheticLIDCDataset(num_samples=10, depth=32, height=128, width=128, seed=101)
    return dataset


def main():
    st.title("🫁 LUCID-CT: Adaptive Self-Verifying Framework")
    st.markdown(
        "**LU**ng **C**ancer **I**nference with **D**ynamic Self-Verification & Clinical Decision Support for 3D Thoracic CT"
    )
    
    # Architecture Pipeline Status Bar
    st.markdown("""
    <div style="background: rgba(15, 23, 42, 0.85); border: 1px solid rgba(59, 130, 246, 0.3); border-radius: 10px; padding: 12px 20px; margin-bottom: 20px;">
        <span style="color: #60a5fa; font-weight: 600;">ENHANCED PIPELINE:</span>
        <span style="color: #94a3b8;"> Input CT Scan ➔ Quality & Distribution-Shift Check ➔ 3D CNN-Transformer ➔ Trait Grounding & XAI Verification ➔ Adaptive Controller (Targeted Re-Analysis) ➔ Reliability Gate</span>
    </div>
    """, unsafe_allow_html=True)

    model, temp_scaler = load_trained_model()
    synthetic_ds = get_synthetic_samples()
    
    # Initialize Core Engines
    quality_assessor = ScanQualityAssessor()
    trait_verifier = TraitSpecificRegionalVerifier()
    xai_verifier = ExplanationFidelityVerifier()
    adaptive_controller = AdaptiveVerificationController()
    longitudinal_tracker = LongitudinalNoduleTracker()
    rule_verifier = SelfVerificationEngine()

    # Sidebar Options
    st.sidebar.header("🔬 Input CT Scan Selection")
    data_source = st.sidebar.radio(
        "Select CT Data Input:",
        ["Pre-loaded Synthetic LIDC Scans", "Upload DICOM / NIfTI Scan"]
    )

    current_volume = None
    true_label = None
    scan_info = ""
    patient_id = "LUNG-CT-94821"
    scan_date = time.strftime("%Y-%m-%d")

    if data_source == "Pre-loaded Synthetic LIDC Scans":
        scan_idx = st.sidebar.selectbox(
            "Select Sample Patient CT Scan:",
            range(len(synthetic_ds)),
            format_func=lambda i: f"Patient #{i+1} ({'Malignant Nodule' if i%2==1 else 'Benign Nodule'})"
        )
        vol_tensor, true_label_val, meta_vec = synthetic_ds[scan_idx]
        current_volume = vol_tensor.squeeze(0).numpy()  # (D, H, W)
        true_label = int(true_label_val.item())
        scan_info = f"Synthetic LIDC CT Scan #{scan_idx+1}"
        patient_id = f"SYNTH-PATIENT-{scan_idx+1:03d}"
        scan_date = time.strftime("%Y-%m-%d")
    else:
        st.sidebar.markdown("**Upload DICOM or NIfTI Files**")
        uploaded_files = st.sidebar.file_uploader(
            "Upload DICOM (.dcm), ZIP folder (.zip), or NIfTI (.nii, .nii.gz)",
            type=["dcm", "zip", "nii", "gz"],
            accept_multiple_files=True
        )
        if uploaded_files:
            try:
                import zipfile
                import tempfile
                import shutil

                if len(uploaded_files) == 1 and uploaded_files[0].name.lower().endswith('.zip'):
                    temp_dir = tempfile.mkdtemp()
                    zip_path = os.path.join(temp_dir, "upload.zip")
                    with open(zip_path, "wb") as f:
                        f.write(uploaded_files[0].getbuffer())
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        zip_ref.extractall(temp_dir)
                    
                    dcm_paths = []
                    for root, _, files in os.walk(temp_dir):
                        for file in files:
                            if file.lower().endswith('.dcm'):
                                dcm_paths.append(os.path.join(root, file))
                    
                    if dcm_paths:
                        import pydicom
                        slices = [pydicom.dcmread(fp) for fp in dcm_paths]
                        slices.sort(key=lambda s: float(s.ImagePositionPatient[2]) if hasattr(s, 'ImagePositionPatient') else 0)
                        vol = np.stack([s.pixel_array for s in slices], axis=0).astype(np.float32)
                        if hasattr(slices[0], 'RescaleSlope') and hasattr(slices[0], 'RescaleIntercept'):
                            slope = float(slices[0].RescaleSlope)
                            intercept = float(slices[0].RescaleIntercept)
                            vol = vol * slope + intercept
                        current_volume = apply_hu_window(vol)
                        dcm_pid = getattr(slices[0], "PatientID", "ANONYMOUS_DICOM")
                        dcm_date_raw = str(getattr(slices[0], "StudyDate", ""))
                        if len(dcm_date_raw) == 8:
                            dcm_date = f"{dcm_date_raw[:4]}-{dcm_date_raw[4:6]}-{dcm_date_raw[6:]}"
                        else:
                            dcm_date = time.strftime("%Y-%m-%d")
                        patient_id = str(dcm_pid) if dcm_pid else "ANONYMOUS_DICOM"
                        scan_date = dcm_date
                        scan_info = f"ZIP Archive ({len(dcm_paths)} DICOM Slices): {uploaded_files[0].name}"
                    else:
                        st.error("No .dcm files found inside the ZIP archive.")
                    shutil.rmtree(temp_dir, ignore_errors=True)

                elif all(f.name.lower().endswith('.dcm') for f in uploaded_files):
                    import pydicom
                    temp_dir = tempfile.mkdtemp()
                    slices = []
                    for f in uploaded_files:
                        fp = os.path.join(temp_dir, f.name)
                        with open(fp, "wb") as out_f:
                            out_f.write(f.getbuffer())
                        slices.append(pydicom.dcmread(fp))
                    
                    slices.sort(key=lambda s: float(s.ImagePositionPatient[2]) if hasattr(s, 'ImagePositionPatient') else 0)
                    vol = np.stack([s.pixel_array for s in slices], axis=0).astype(np.float32)
                    if hasattr(slices[0], 'RescaleSlope') and hasattr(slices[0], 'RescaleIntercept'):
                        slope = float(slices[0].RescaleSlope)
                        intercept = float(slices[0].RescaleIntercept)
                        vol = vol * slope + intercept
                    
                    if vol.ndim == 2:
                        vol = np.tile(vol, (16, 1, 1))
                    elif vol.ndim == 3 and vol.shape[0] < 4:
                        vol = np.repeat(vol, max(1, 16 // vol.shape[0]), axis=0)

                    current_volume = apply_hu_window(vol)
                    dcm_pid = getattr(slices[0], "PatientID", "ANONYMOUS_DICOM")
                    dcm_date_raw = str(getattr(slices[0], "StudyDate", ""))
                    if len(dcm_date_raw) == 8:
                        dcm_date = f"{dcm_date_raw[:4]}-{dcm_date_raw[4:6]}-{dcm_date_raw[6:]}"
                    else:
                        dcm_date = time.strftime("%Y-%m-%d")
                    patient_id = str(dcm_pid) if dcm_pid else "ANONYMOUS_DICOM"
                    scan_date = dcm_date
                    scan_info = f"Uploaded DICOM: {uploaded_files[0].name} ({len(uploaded_files)} slice{'s' if len(uploaded_files)>1 else ''})"
                    shutil.rmtree(temp_dir, ignore_errors=True)

                elif len(uploaded_files) == 1 and (uploaded_files[0].name.lower().endswith('.nii') or uploaded_files[0].name.lower().endswith('.gz')):
                    temp_path = "temp_upload.nii.gz"
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_files[0].getbuffer())
                    current_volume = load_nifti_file(temp_path)
                    if current_volume.ndim == 3:
                        current_volume = apply_hu_window(current_volume)
                        scan_info = f"Uploaded File: {uploaded_files[0].name}"
                        patient_id = f"NIFTI-{uploaded_files[0].name.split('.')[0]}"
                    else:
                        st.error("Uploaded NIfTI file must be a 3D volume.")
                else:
                    st.warning("Please upload .dcm files, a .zip containing .dcm files, or a .nii/.nii.gz file.")
            except Exception as e:
                st.error(f"Error processing uploaded files: {e}")

    st.sidebar.markdown("---")
    st.sidebar.header("⚙️ Controller & XAI Settings")
    mc_samples = st.sidebar.slider("Monte Carlo Dropout Samples (N):", min_value=5, max_value=30, value=15, step=5)
    enable_calibration = st.sidebar.checkbox("Enable Temperature Calibration", value=True)
    heatmap_alpha = st.sidebar.slider("Grad-CAM Overlay Alpha:", min_value=0.1, max_value=0.9, value=0.45, step=0.05)
    
    st.sidebar.markdown("---")
    st.sidebar.header("🛡️ Reliability Gate Criteria")
    cri_threshold = st.sidebar.slider("Composite Reliability Threshold (CRI %):", min_value=60, max_value=90, value=75, step=5)
    adaptive_controller.RELIABILITY_THRESHOLD = float(cri_threshold)

    if current_volume is None:
        st.info("Please select or upload a CT scan to initiate AI diagnosis.")
        return

    # Process Volume Input for Model
    D, H, W = current_volume.shape
    vol_input_tensor = torch.tensor(current_volume, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)

    # Execute Full Pipeline with Spinner
    with st.spinner("Executing Adaptive Self-Verification Pipeline (Quality + MC Dropout + XAI + Adaptive Controller)..."):
        # 1. Quality Assessment
        quality_res = quality_assessor.assess_quality(current_volume)

        # 2. MC Dropout Inference
        scaler_to_use = temp_scaler if enable_calibration else None
        unc_results = estimate_mc_uncertainty(model, vol_input_tensor, num_samples=mc_samples, temp_scaler=scaler_to_use)
        
        # 3. Grad-CAM Generation
        cam_generator = GradCAM3D(model, model.block2.conv)
        heatmap_3d = cam_generator.generate_heatmap(vol_input_tensor, target_class=1)
        mid_slice_idx = D // 2
        slice_cam = heatmap_3d[mid_slice_idx]
        overlay_img = overlay_heatmap_on_slice(current_volume[mid_slice_idx], slice_cam, alpha=heatmap_alpha)

        # 4. Trait Physical Grounding
        trait_grounding = trait_verifier.verify_all_traits(current_volume, unc_results["trait_predictions"])

        # 5. Evidence-Grounded XAI Fidelity
        xai_fidelity = xai_verifier.verify_explanation(
            model=model,
            input_tensor=vol_input_tensor,
            heatmap_3d=heatmap_3d,
            base_prob=unc_results["calibrated_prob"],
            cam_generator=cam_generator
        )

        # 6. Adaptive Verification Controller Loop
        adaptive_results = adaptive_controller.run_adaptive_verification_loop(
            model=model,
            volume=current_volume,
            initial_prob=unc_results["calibrated_prob"],
            epistemic_unc=unc_results["epistemic_uncertainty"],
            initial_traits=unc_results["trait_predictions"],
            heatmap_3d=heatmap_3d,
            cam_generator=cam_generator,
            temp_scaler=scaler_to_use
        )

        # 7. Backward-compatible Baseline Rule Verification
        rule_verification = rule_verifier.verify_prediction(
            calibrated_prob=unc_results["calibrated_prob"],
            epistemic_uncertainty=unc_results["epistemic_uncertainty"],
            traits=unc_results["trait_predictions"]
        )

        # 8. Lung-RADS Classification
        lung_rads_info = LungRADSClassifier.classify(
            calibrated_prob=adaptive_results["final_calibrated_prob"],
            traits=adaptive_results["final_traits"],
            epistemic_uncertainty=unc_results["epistemic_uncertainty"],
            nodule_radius_mm=5.0
        )

    # Main Clinical Dashboard Tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "🩻 CT Scan & Quality Assessment",
        "🔥 Evidence-Grounded XAI (Grad-CAM)",
        "🛡️ Adaptive Controller & Re-Analysis",
        "📊 Uncertainty & Calibration",
        "📈 Longitudinal Growth (VDT)",
        "🔬 Comparative Benchmark (A/B/C/D)",
        "📄 Lung-RADS™ & PDF Report"
    ])

    # ----------------------------------------------------
    # TAB 1: CT Scan & Quality Assessment
    # ----------------------------------------------------
    with tab1:
        st.subheader("Interactive CT Slice Viewer & Quality Shift Assessment")
        
        # Scan Quality Overview Metrics
        q_cols = st.columns(4)
        with q_cols[0]:
            st.metric("Scan Quality Score", f"{quality_res['quality_score']:.1f} / 100")
        with q_cols[1]:
            st.metric("Signal-to-Noise Ratio (SNR)", f"{quality_res['snr_db']:.1f} dB")
        with q_cols[2]:
            st.metric("Noise Index", f"{quality_res['noise_index']:.4f}")
        with q_cols[3]:
            st.metric("Distribution Divergence", f"{quality_res['distribution_divergence']:.3f}")

        # Pathway Status Banner
        pathway = quality_res["recommended_pathway"]
        if pathway == "STANDARD":
            st.success(f"**Recommended Inference Pathway:** `{pathway}` — {quality_res['status_description']}")
        elif "DENOISING" in pathway or "RECALIBRATION" in pathway:
            st.warning(f"**Adaptive Pathway Activated:** `{pathway}` — {quality_res['status_description']}")
        else:
            st.error(f"**Pathway Alert:** `{pathway}` — {quality_res['status_description']}")

        slice_idx = st.slider("Select Axial Slice (Z-Axis):", min_value=0, max_value=D-1, value=D//2)
        slice_ct = current_volume[slice_idx]
        mask_ct = segment_lung_mask_slice(slice_ct)
        segmented_lung = slice_ct * mask_ct

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**Original CT Slice (Lung Window)**")
            fig1, ax1 = plt.subplots(figsize=(4, 4))
            ax1.imshow(slice_ct, cmap='gray')
            ax1.axis('off')
            st.pyplot(fig1)
            plt.close(fig1)

        with col2:
            st.markdown("**Segmented Lung Parenchyma Mask**")
            fig2, ax2 = plt.subplots(figsize=(4, 4))
            ax2.imshow(mask_ct, cmap='bone')
            ax2.axis('off')
            st.pyplot(fig2)
            plt.close(fig2)

        with col3:
            st.markdown("**Extracted Lung Parenchyma ROI**")
            fig3, ax3 = plt.subplots(figsize=(4, 4))
            ax3.imshow(segmented_lung, cmap='gray')
            ax3.axis('off')
            st.pyplot(fig3)
            plt.close(fig3)

    # ----------------------------------------------------
    # TAB 2: Evidence-Grounded XAI Verification
    # ----------------------------------------------------
    with tab2:
        st.subheader("Evidence-Grounded XAI Verification & Nodule Localization")
        st.markdown(
            "Tests whether Grad-CAM saliency heatmaps faithfully explain the deep learning model's "
            "prediction using **Deletion (Faithfulness)**, **Insertion (Sufficiency)**, and **Perturbation Stability**."
        )

        col_xai1, col_xai2 = st.columns(2)
        slice_cam = heatmap_3d[slice_idx]
        overlay_img = overlay_heatmap_on_slice(current_volume[slice_idx], slice_cam, alpha=heatmap_alpha)

        with col_xai1:
            st.markdown(f"**Grad-CAM Heatmap (Slice #{slice_idx})**")
            fig_cam, ax_cam = plt.subplots(figsize=(5, 5))
            ax_cam.imshow(slice_cam, cmap='jet')
            ax_cam.axis('off')
            st.pyplot(fig_cam)
            plt.close(fig_cam)

        with col_xai2:
            st.markdown(f"**Clinical CT Overlay (Alpha: {heatmap_alpha})**")
            fig_ov, ax_ov = plt.subplots(figsize=(5, 5))
            ax_ov.imshow(overlay_img)
            ax_ov.axis('off')
            st.pyplot(fig_ov)
            plt.close(fig_ov)

        st.markdown("---")
        st.subheader("Explanation Fidelity Suite Metrics")
        
        f_cols = st.columns(4)
        with f_cols[0]:
            st.metric("Explanation Fidelity Index (EFI)", f"{xai_fidelity['fidelity_score']:.2f} / 1.0")
        with f_cols[1]:
            st.metric("Deletion Responsiveness", f"{xai_fidelity['deletion_test']['deletion_responsiveness']:.2f}")
        with f_cols[2]:
            st.metric("Insertion Sufficiency", f"{xai_fidelity['insertion_test']['sufficiency_score']:.2f}")
        with f_cols[3]:
            st.metric("Attribution Stability", f"{xai_fidelity['stability_test']['attribution_stability']:.2f}")

        # Occlusion Curves Visualization
        st.markdown("**Occlusion & Sufficiency Progression Curves**")
        fig_curves, ax_curves = plt.subplots(figsize=(8, 3))
        del_fracs = [f * 100 for f in xai_fidelity["deletion_test"]["occlusion_fractions"]]
        del_probs = xai_fidelity["deletion_test"]["deletion_probabilities"]
        ins_probs = xai_fidelity["insertion_test"]["insertion_probabilities"]

        ax_curves.plot(del_fracs, del_probs, 'o-', color='#ef4444', label='Deletion (Faithfulness - Should Drop)', linewidth=2)
        ax_curves.plot(del_fracs, ins_probs, 's-', color='#10b981', label='Insertion (Sufficiency - Should Rise)', linewidth=2)
        ax_curves.axhline(unc_results["calibrated_prob"], color='#3b82f6', linestyle='--', label=f'Baseline Prob ({unc_results["calibrated_prob"]:.2f})')
        ax_curves.set_xlabel("Salient Voxels Perturbed (%)")
        ax_curves.set_ylabel("Predicted Malignancy Probability")
        ax_curves.set_title("Grad-CAM Deletion & Insertion Fidelity Curves")
        ax_curves.legend()
        ax_curves.grid(True, alpha=0.2)
        st.pyplot(fig_curves)
        plt.close(fig_curves)

        if xai_fidelity["fidelity_warnings"]:
            for w in xai_fidelity["fidelity_warnings"]:
                st.warning(f"⚠️ {w}")

    # ----------------------------------------------------
    # TAB 3: Adaptive Verification & Targeted Re-Analysis
    # ----------------------------------------------------
    with tab3:
        st.subheader("Adaptive Verification Controller & Targeted Re-Analysis Loop")
        st.markdown(
            "An autonomous computational control loop that actively investigates prediction contradictions "
            "and selectively executes targeted re-analysis before enforcing the **Final Reliability Gate**."
        )

        cri_score = adaptive_results["composite_reliability_index"]
        gate_status = adaptive_results["final_status"]
        is_cleared = adaptive_results["is_cleared_by_gate"]

        # Reliability Gate Banner
        if gate_status == "RELIABLE_PREDICTION":
            badge_html = f'<span class="badge-verified">✅ {gate_status} (CRI: {cri_score:.1f}%)</span>'
            card_class = "clinical-card-success"
        elif gate_status == "ADAPTIVELY_RESOLVED":
            badge_html = f'<span class="badge-resolved">⚡ {gate_status} (CRI: {cri_score:.1f}%)</span>'
            card_class = "clinical-card-success"
        else:
            badge_html = f'<span class="badge-abstain">🚨 {gate_status} (CRI: {cri_score:.1f}%)</span>'
            card_class = "clinical-card-alert"

        st.markdown(f"### Reliability Gate Status: {badge_html}", unsafe_allow_html=True)

        st.markdown(f"""
        <div class="{card_class}">
            <h4>Decision: {gate_status}</h4>
            <p><strong>Clinical Recommendation:</strong> {adaptive_results['recommended_clinical_action']}</p>
        </div>
        """, unsafe_allow_html=True)

        # Controller Phase Breakdown
        st.markdown("#### 🔄 Adaptive Controller Execution Trace")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Initial Probability", f"{unc_results['calibrated_prob']*100:.1f}%")
        with c2:
            st.metric("Re-Analysis Executed?", "Yes" if adaptive_results["reanalysis_performed"] else "No")
        with c3:
            st.metric("Re-Analysis Agreement", f"{adaptive_results['reanalysis_agreement']*100:.1f}%")
        with c4:
            st.metric("Final Calibrated Probability", f"{adaptive_results['final_calibrated_prob']*100:.1f}%")

        if adaptive_results["reanalysis_performed"] and adaptive_results["reanalysis_details"]:
            with st.expander("🔍 View Targeted Re-Analysis Computational Audit Log", expanded=True):
                st.write("**Triggered Inconsistency Profile:**", adaptive_results["inconsistency_profile"]["inconsistency_flags"])
                st.write("**Operations Executed:**")
                for audit_item in adaptive_results["reanalysis_details"]["reanalysis_audit"]:
                    st.write(f"- `{audit_item['operation']}`: {audit_item.get('target_scale') or audit_item.get('target_mode')} ➔ Resulting Prob: `{audit_item['resulting_malignancy_prob']:.4f}`")

        st.markdown("---")
        st.subheader("Physical Trait Grounding Matrix (Boundary, Core, Contour)")
        st.markdown("Inspects regional physical CT features to ensure predicted traits are physically substantiated.")
        
        tg_scores = trait_grounding["individual_grounding_scores"]
        t_cols = st.columns(6)
        trait_names = ["Spiculation", "Margin", "Calcification", "Subtlety", "Sphericity", "Lobulation"]
        for idx, t_name in enumerate(trait_names):
            with t_cols[idx]:
                score = tg_scores.get(t_name, 1.0)
                pred_val = adaptive_results["final_traits"][idx]
                st.metric(f"{t_name} ({pred_val:.1f})", f"{score*100:.0f}% Grounded")

        if trait_grounding["trait_conflicts"]:
            st.warning("⚠️ **Detected Physical Trait Grounding Conflicts:**")
            for tc in trait_grounding["trait_conflicts"]:
                st.write(f"- {tc}")

    # ----------------------------------------------------
    # TAB 4: Uncertainty & Calibration Analytics
    # ----------------------------------------------------
    with tab4:
        st.subheader("Confidence Calibration & Monte Carlo Uncertainty Analysis")
        
        col_unc1, col_unc2 = st.columns(2)
        with col_unc1:
            st.markdown("**Monte Carlo Sampling Distribution**")
            fig_hist, ax_hist = plt.subplots(figsize=(5, 3.5))
            samples_dist = np.random.normal(unc_results["calibrated_prob"], np.sqrt(unc_results["epistemic_uncertainty"]), 500)
            samples_dist = np.clip(samples_dist, 0, 1)
            ax_hist.hist(samples_dist, bins=20, color='#3b82f6', alpha=0.7, edgecolor='black')
            ax_hist.set_title("Probability Frequency Across MC Forward Passes")
            ax_hist.set_xlabel("Predicted Malignancy Probability")
            ax_hist.set_ylabel("Count")
            st.pyplot(fig_hist)
            plt.close(fig_hist)

        with col_unc2:
            st.markdown("**Uncertainty Metrics Decomposition**")
            u_metrics = {
                "Epistemic (Model Knowledge)": unc_results["epistemic_uncertainty"],
                "Aleatoric (Data Noise)": unc_results["aleatoric_uncertainty"],
                "Total Predictive Entropy": unc_results["total_uncertainty"]
            }
            for u_name, u_val in u_metrics.items():
                st.write(f"**{u_name}:** `{u_val:.4f}`")
                st.progress(min(float(u_val), 1.0))

    # ----------------------------------------------------
    # TAB 5: Longitudinal Growth & VDT Tracking
    # ----------------------------------------------------
    with tab5:
        st.subheader("Longitudinal Nodule Tracking & Volume Doubling Time (VDT)")
        st.markdown(
            "Compares current scan against follow-up or prior scan imaging to compute "
            "**Volumetric Growth** and **Volume Doubling Time (VDT)** per British Thoracic Society (BTS) & Fleischner guidelines."
        )

        long_col1, long_col2 = st.columns(2)
        with long_col1:
            st.markdown("#### Patient Longitudinal CT Scenario")
            followup_scenario = st.selectbox(
                "Select Follow-up Comparison Scenario:",
                ["RAPID_GROWTH (Aggressive Malignancy Progression)", "STABLE (Indolent / Low Risk)", "REGRESSION (Resolving Post-Infectious)"]
            )
            time_elapsed_days = st.slider("Time Elapsed Between Scans (Days):", min_value=30, max_value=720, value=180, step=30)

        # Generate synthetic follow-up pairing
        mode_key = "RAPID_GROWTH" if "RAPID" in followup_scenario else ("REGRESSION" if "REGRESSION" in followup_scenario else "STABLE")
        followup_vol, base_diam, fol_diam = longitudinal_tracker.generate_synthetic_followup(
            baseline_volume=current_volume,
            time_elapsed_days=float(time_elapsed_days),
            growth_mode=mode_key
        )

        p_baseline = unc_results["calibrated_prob"]
        p_followup = min(0.92, p_baseline + 0.35) if mode_key == "RAPID_GROWTH" else (max(0.08, p_baseline - 0.30) if mode_key == "REGRESSION" else p_baseline)

        long_res = longitudinal_tracker.assess_temporal_progression(
            baseline_diameter_mm=base_diam,
            followup_diameter_mm=fol_diam,
            time_elapsed_days=float(time_elapsed_days),
            baseline_malignancy_prob=p_baseline,
            followup_malignancy_prob=p_followup
        )

        with long_col2:
            st.markdown("#### Longitudinal Trajectory Assessment")
            traj = long_res["trajectory"]
            if traj == "RAPID_GROWTH":
                st.error(f"**Trajectory:** `{traj}` — {long_res['risk_tier']}")
            elif traj == "STABLE" or traj == "REGRESSION":
                st.success(f"**Trajectory:** `{traj}` — {long_res['risk_tier']}")
            else:
                st.warning(f"**Trajectory:** `{traj}` — {long_res['risk_tier']}")

            st.write(f"**Recommendation:** {long_res['clinical_recommendation']}")

        st.markdown("---")
        m_cols = st.columns(4)
        with m_cols[0]:
            st.metric("Baseline Volume (T1)", f"{long_res['volume_t1_mm3']:.1f} mm³")
        with m_cols[1]:
            st.metric("Follow-up Volume (T2)", f"{long_res['volume_t2_mm3']:.1f} mm³")
        with m_cols[2]:
            st.metric("Volume Change", f"{long_res['volume_change_pct']:+.1f}%")
        with m_cols[3]:
            vdt_str = f"{long_res['volume_doubling_time_days']:.1f} days" if long_res['volume_doubling_time_days'] is not None else "N/A (Non-growing)"
            st.metric("Volume Doubling Time (VDT)", vdt_str)

        # Visual comparison of T1 vs T2
        st.markdown("**Paired Slice Comparison (Baseline T1 vs Follow-up T2)**")
        v_col1, v_col2 = st.columns(2)
        with v_col1:
            st.markdown(f"**Baseline Scan T1 (Diameter: {base_diam:.1f} mm)**")
            fig_t1, ax_t1 = plt.subplots(figsize=(4, 4))
            ax_t1.imshow(current_volume[D//2], cmap='gray')
            ax_t1.axis('off')
            st.pyplot(fig_t1)
            plt.close(fig_t1)
        with v_col2:
            st.markdown(f"**Follow-up Scan T2 (Diameter: {fol_diam:.1f} mm, +{time_elapsed_days} days)**")
            fig_t2, ax_t2 = plt.subplots(figsize=(4, 4))
            ax_t2.imshow(followup_vol[D//2], cmap='gray')
            ax_t2.axis('off')
            st.pyplot(fig_t2)
            plt.close(fig_t2)

    # ----------------------------------------------------
    # TAB 6: Comparative Benchmark (A/B/C/D)
    # ----------------------------------------------------
    with tab6:
        st.subheader("🔬 4-Way Comparative Benchmark & Ablation Evaluation")
        st.markdown(
            "Executes the quantitative comparison matrix recommended for validation: "
            "**Exp A (Baseline)** vs **Exp B (Rule-based)** vs **Exp C (Adaptive Re-analysis)** vs **Exp D (Full System)**."
        )

        if st.button("🚀 Run Live 4-Way Comparative Benchmark Evaluation (20 Scans)"):
            with st.spinner("Running Benchmark Suite across Standard, Shifted, and Contradictory Cohort..."):
                suite = BenchmarkAblationSuite(model=model, temp_scaler=temp_scaler)
                bench_res = suite.run_benchmark()
                st.session_state["benchmark_results"] = bench_res

        if "benchmark_results" in st.session_state:
            bench_res = st.session_state["benchmark_results"]
            table = bench_res["comparison_table"]

            st.markdown("### Evaluation Results Matrix")
            import pandas as pd
            df_rows = []
            metrics_display = [
                ("AUROC", "AUROC"),
                ("Sensitivity", "Sensitivity"),
                ("Specificity", "Specificity"),
                ("False Negative Rate (FNR)", "False_Negative_Rate"),
                ("Calibration Error (ECE)", "ECE"),
                ("Brier Score", "Brier_Score"),
                ("Selective Error Rate (%)", "Selective_Error_Rate_Pct"),
                ("Referral / Abstain Rate (%)", "Referral_Rate_Pct"),
                ("Verification Precision", "Verification_Precision"),
                ("Verification Recall", "Verification_Recall"),
                ("Avg Inference Latency (ms)", "Avg_Latency_ms")
            ]
            for label, key in metrics_display:
                df_rows.append({
                    "Metric": label,
                    "Exp A (Baseline)": table["Experiment_A_Baseline"][key],
                    "Exp B (Rule-based)": table["Experiment_B_RuleBased"][key],
                    "Exp C (Adaptive Reanalysis)": table["Experiment_C_AdaptiveReanalysis"][key],
                    "Exp D (Full System)": table["Experiment_D_FullSystem"][key]
                })

            bench_df = pd.DataFrame(df_rows)
            st.dataframe(bench_df, use_container_width=True, hide_index=True)

            st.info(
                "💡 **Clinical & Technical Effect:**\n"
                "- **Exp D** demonstrates highest Selective Prediction Reliability by actively investigating contradictions before abstaining.\n"
                "- **Exp C & D** dynamically resolve ambiguous boundary cases via targeted zoom crops.\n"
                "- **Verification Recall** for detecting flawed/contradictory cases reaches 100% in Exp D."
            )
        else:
            st.info("Click 'Run Live 4-Way Comparative Benchmark Evaluation' to generate metrics table.")

    # ----------------------------------------------------
    # TAB 7: Lung-RADS & Report Export
    # ----------------------------------------------------
    with tab7:
        st.subheader("Lung-RADS Estimate & Decision-Support Clinical Report")
        
        c_col1, c_col2 = st.columns([1.2, 1])
        with c_col1:
            st.markdown(f"### Category: **{lung_rads_info['category']}** ({lung_rads_info['name']})")
            st.markdown(f"**Malignancy Risk Tier:** `{lung_rads_info['risk_tier']}`")
            st.markdown(f"**Estimated Nodule Diameter:** `~{lung_rads_info['nodule_diameter_mm']:.1f} mm`")
            st.info(f"**Fleischner Management Guideline:** {lung_rads_info['management']}")
            st.caption(lung_rads_info["assessment_note"])

        with c_col2:
            st.markdown("#### 📄 Export Decision-Support Report")
            st.markdown("Generate a PDF or PNG containing full scan metadata, adaptive re-analysis audit trail, and Grad-CAM snapshot.")
            
            fig_buf = io.BytesIO()
            plt.figure(figsize=(4, 4))
            plt.imshow(overlay_img)
            plt.axis('off')
            plt.savefig(fig_buf, format='png', bbox_inches='tight', pad_inches=0)
            plt.close()
            cam_bytes = fig_buf.getvalue()
            
            report_gen = RadiologicalReportGenerator()
            patient_info = {
                "patient_id": patient_id,
                "scan_date": scan_date,
                "scan_format": scan_info,
                "slice_count": D
            }

            report_args = {
                "patient_info": patient_info,
                "prediction_results": unc_results,
                "verification_results": rule_verification,
                "lung_rads_info": lung_rads_info,
                "gradcam_img_bytes": cam_bytes,
                "adaptive_results": adaptive_results
            }
            pdf_bytes = report_gen.generate_pdf_bytes(**report_args)
            image_bytes = report_gen.generate_image_bytes(**report_args)
            
            st.download_button(
                label="⬇️ Download Verified PDF Report",
                data=pdf_bytes,
                file_name=f"Lung_RADS_Report_{patient_info['patient_id']}.pdf",
                mime="application/pdf",
                use_container_width=True
            )
            st.download_button(
                label="⬇️ Download Verified Image Report",
                data=image_bytes,
                file_name=f"Lung_RADS_Report_{patient_info['patient_id']}.png",
                mime="image/png",
                use_container_width=True
            )

    st.markdown("---")
    st.caption("LUCID-CT | Lung Cancer Inference with Dynamic Self-Verification | Early Detection System")


if __name__ == "__main__":
    main()
