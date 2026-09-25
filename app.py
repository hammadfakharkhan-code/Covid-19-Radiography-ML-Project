import os
import io
import time
import numpy as np
import pandas as pd
from PIL import Image
import joblib
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title=" | COVID-19 Diagnostic Suite",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------
# Custom Styling (Clinical Medical  Theme)
# ---------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background: radial-gradient(circle at 10% 10%, #0d1527 0%, #060913 100%);
        color: #e2e8f0;
    }
    
    /* Header Banner */
    .clinical-banner {
        background: linear-gradient(135deg, rgba(30, 58, 138, 0.4) 0%, rgba(15, 23, 42, 0.7) 100%);
        border: 1px solid rgba(59, 130, 246, 0.25);
        border-radius: 12px;
        padding: 22px 28px;
        margin-bottom: 24px;
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.35);
    }
    
    .clinical-title {
        font-size: 26px;
        font-weight: 700;
        color: #60a5fa;
        margin-bottom: 6px;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    
    .clinical-subtitle {
        color: #94a3b8;
        font-size: 14px;
        margin: 0;
    }
    
    /* Diagnostic Result Cards */
    .result-card-covid {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.15) 0%, rgba(127, 29, 29, 0.25) 100%);
        border: 1px solid #ef4444;
        border-radius: 14px;
        padding: 24px;
        text-align: center;
        box-shadow: 0 0 25px rgba(239, 68, 68, 0.2);
    }
    
    .result-card-normal {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(6, 78, 59, 0.25) 100%);
        border: 1px solid #10b981;
        border-radius: 14px;
        padding: 24px;
        text-align: center;
        box-shadow: 0 0 25px rgba(16, 185, 129, 0.2);
    }
    
    .badge-covid {
        background-color: #ef4444;
        color: white;
        padding: 6px 16px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 18px;
        letter-spacing: 0.5px;
        display: inline-block;
        margin-bottom: 12px;
    }
    
    .badge-normal {
        background-color: #10b981;
        color: white;
        padding: 6px 16px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 18px;
        letter-spacing: 0.5px;
        display: inline-block;
        margin-bottom: 12px;
    }

    /* Metric Cards */
    .stat-box {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(148, 163, 184, 0.15);
        border-radius: 10px;
        padding: 16px;
        text-align: center;
    }
    .stat-val {
        font-size: 28px;
        font-weight: 700;
        color: #38bdf8;
    }
    .stat-lbl {
        font-size: 12px;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Load Model Pipeline
# ---------------------------------------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "pipeline.joblib")

@st.cache_resource(show_spinner="Loading trained  model pipeline...")
def load_pipeline():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)

pipeline = load_pipeline()

# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/lungs.png", width=64)
    st.title("")
    st.caption(" Covid-19 Radiograph Bagging-SVC Classifier")
    st.divider()

    selected_tab = st.radio(
        "Navigation",
        ["🔍 Radiograph Diagnosis", "📊 Model Performance & PCA", "ℹ️ System & Architecture"],
        index=0
    )
    st.divider()

    st.markdown("### ⚙️ Pipeline Specifications")
    st.markdown("""
    - **Input**: 64×64 Grayscale Radiograph (4,096 px)
    - **Scaler**: `StandardScaler` (Z-score)
    - **Latent Dim**: `PCA` (100 components)
    - **Classifier**: `BaggingClassifier(SVC)`
    - **Sub-estimators**: 100 SVMs (RBF kernel)
    - **Sampling**: 50% bootstrap per estimator
    """)
    st.divider()
    st.caption("Covid-19 Diagnostic Research Prototype by Hammad Fakhar • Email: hammadfakharkhan@gmail.com • For any query contact through mail.   ")

# ---------------------------------------------------------
# Main App Header
# ---------------------------------------------------------
st.markdown("""
<div class="clinical-banner">
    <div class="clinical-title">
        <span>🫁</span>
        <span> — COVID-19 Radiograph Diagnostic Suite</span>
    </div>
    <div class="clinical-subtitle">
        High-throughput Computer-Aided Diagnosis (CAD) powered by PCA Dimensionality Reduction & Bagged Support Vector Ensembles.
    </div>
</div>
""", unsafe_allow_html=True)

if pipeline is None:
    st.error("⚠️ Trained model pipeline not found (`pipeline.joblib`).")
    st.info("The model is currently training or needs to be generated.")
    if st.button("🚀 Start Training Pipeline Now"):
        with st.spinner("Training model on dataset... please check terminal logs."):
            import subprocess
            res = subprocess.run(["python", os.path.join(os.path.dirname(__file__), "train_model.py")], capture_output=True, text=True)
            if res.returncode == 0:
                st.success("Model trained and pipeline.joblib generated successfully! Reloading...")
                st.rerun()
            else:
                st.error(f"Training failed: {res.stderr}")
    st.stop()

scaler = pipeline['scaler']
pca = pipeline['pca']
model = pipeline['model']
metrics = pipeline['metrics']
classes = pipeline['classes']
sample_gallery = pipeline.get('sample_gallery', [])

# ---------------------------------------------------------
# TAB 1: RADIOGRAPH DIAGNOSIS
# ---------------------------------------------------------
if selected_tab == "🔍 Radiograph Diagnosis":
    st.subheader("1. Patient Radiograph Input")

    col_input, col_preset = st.columns([3, 2], gap="large")

    selected_image_obj = None
    image_source_name = ""

    with col_input:
        uploaded_file = st.file_uploader(
            "Upload Chest X-Ray scan (DICOM export / PNG / JPG)",
            type=["png", "jpg", "jpeg"],
            help="Drop any frontal chest radiograph image."
        )
        if uploaded_file is not None:
            selected_image_obj = Image.open(uploaded_file)
            image_source_name = uploaded_file.name

    with col_preset:
        st.markdown("**Or pick a preloaded clinical benchmark sample:**")
        if sample_gallery:
            covid_samples = [s for s in sample_gallery if s['label'] == 'COVID-19'][:3]
            normal_samples = [s for s in sample_gallery if s['label'] != 'COVID-19'][:3]

            col_c, col_n = st.columns(2)
            with col_c:
                st.markdown("<span style='color:#f87171; font-weight:600;'>COVID-19 Samples</span>", unsafe_allow_html=True)
                for s in covid_samples:
                    if st.button(f"🔴 {s['filename']}", key=s['key'], use_container_width=True):
                        if os.path.exists(s['path']):
                            selected_image_obj = Image.open(s['path'])
                            image_source_name = f"Sample: {s['filename']}"
            with col_n:
                st.markdown("<span style='color:#34d399; font-weight:600;'>Normal Samples</span>", unsafe_allow_html=True)
                for s in normal_samples:
                    if st.button(f"🟢 {s['filename']}", key=s['key'], use_container_width=True):
                        if os.path.exists(s['path']):
                            selected_image_obj = Image.open(s['path'])
                            image_source_name = f"Sample: {s['filename']}"

    # Default fallback to first sample if nothing selected
    if selected_image_obj is None and sample_gallery:
        first_sample = sample_gallery[0]
        if os.path.exists(first_sample['path']):
            selected_image_obj = Image.open(first_sample['path'])
            image_source_name = f"Sample: {first_sample['filename']} (Default Preview)"

    if selected_image_obj is not None:
        st.divider()
        st.markdown(f"#### Analyzing: `{image_source_name}`")

        # Preprocessing
        img_gray = selected_image_obj.convert('L').resize((64, 64))
        raw_pixels = np.array(img_gray, dtype=np.float32).flatten().reshape(1, -1)

        # Scale and PCA
        scaled_vec = scaler.transform(raw_pixels)
        pca_vec = pca.transform(scaled_vec)

        # PCA Reconstruction (to visualize what the model sees after 100 components)
        reconstructed_scaled = pca.inverse_transform(pca_vec)
        reconstructed_raw = scaler.inverse_transform(reconstructed_scaled)
        reconstructed_img = np.clip(reconstructed_raw.reshape(64, 64), 0, 255).astype(np.uint8)

        # Predict
        prediction = model.predict(pca_vec)[0]
        probabilities = model.predict_proba(pca_vec)[0]
        covid_idx = classes.index('covid')
        normal_idx = classes.index('non_covid')

        covid_prob = probabilities[covid_idx]
        normal_prob = probabilities[normal_idx]

        # ----------------- Visual Inspection Row -----------------
        col_view1, col_view2, col_view3 = st.columns(3)
        with col_view1:
            st.markdown("**Original Chest Radiograph**")
            st.image(selected_image_obj, use_container_width=True)
            st.caption(f"Input Resolution: {selected_image_obj.size[0]}×{selected_image_obj.size[1]} px")

        with col_view2:
            st.markdown("**Model Input (64×64 Grayscale)**")
            st.image(img_gray, use_container_width=True)
            st.caption("Standardized to 4,096 pixel intensity values")

        with col_view3:
            st.markdown("**PCA 100-Component Reconstruction**")
            st.image(reconstructed_img, use_container_width=True)
            st.caption("Reconstructed from 100 latent Eigen-components")

        st.divider()

        # ----------------- Diagnostic Verdict Row -----------------
        verdict_col, explain_col = st.columns([3, 4], gap="large")

        with verdict_col:
            st.markdown("### Clinical Diagnostic Output")
            if prediction == 'covid':
                st.markdown(f"""
                <div class="result-card-covid">
                    <div class="badge-covid">⚠️ COVID-19 POSITIVE DETECTED</div>
                    <div style="font-size: 15px; color: #fecaca; margin-bottom: 12px;">
                        Radiographic patterns strongly correlate with viral pneumonic consolidation and ground-glass opacities.
                    </div>
                    <div style="font-size: 32px; font-weight: 800; color: #ef4444;">
                        {covid_prob * 100:.1f}% Confidence
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-card-normal">
                    <div class="badge-normal">✅ NORMAL / NON-COVID</div>
                    <div style="font-size: 15px; color: #a7f3d0; margin-bottom: 12px;">
                        No distinctive COVID-19 peripheral bilateral infiltrates detected in the latent feature space.
                    </div>
                    <div style="font-size: 32px; font-weight: 800; color: #10b981;">
                        {normal_prob * 100:.1f}% Confidence
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("**Diagnostic Probability Distribution:**")
            prob_df = pd.DataFrame({
                "Condition": ["COVID-19 Positive", "Normal (Non-COVID)"],
                "Probability": [covid_prob, normal_prob]
            })
            fig_prob = px.bar(
                prob_df,
                x="Probability",
                y="Condition",
                orientation='h',
                color="Condition",
                color_discrete_map={
                    "COVID-19 Positive": "#ef4444",
                    "Normal (Non-COVID)": "#10b981"
                },
                range_x=[0, 1],
                text_auto='.1%'
            )
            fig_prob.update_layout(
                showlegend=False,
                margin=dict(l=10, r=10, t=10, b=10),
                height=160,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color="#cbd5e1")
            )
            st.plotly_chart(fig_prob, use_container_width=True)

        with explain_col:
            st.markdown("### Latent Feature Explainability (Top PCA Loadings)")
            st.caption("Component weights extracted by PCA for this patient's radiograph:")
            
            top_pcs = pd.DataFrame({
                "Principal Component": [f"PC-{i+1}" for i in range(12)],
                "Latent Activation": pca_vec[0][:12]
            })
            fig_pc = px.bar(
                top_pcs,
                x="Principal Component",
                y="Latent Activation",
                color="Latent Activation",
                color_continuous_scale="Viridis"
            )
            fig_pc.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                height=260,
                margin=dict(l=10, r=10, t=20, b=10),
                font=dict(color="#94a3b8")
            )
            st.plotly_chart(fig_pc, use_container_width=True)

            st.info("""
            **Medical Advisory Notice:**
            This model is generated using Principal Component Analysis and Bagged Support Vector Machines. It is intended strictly as an investigative decision-support tool. Final diagnosis should be verified via RT-PCR or clinical chest CT findings.
            """)

# ---------------------------------------------------------
# TAB 2: MODEL PERFORMANCE & PCA
# ---------------------------------------------------------
elif selected_tab == "📊 Model Performance & PCA":
    st.subheader("Model Evaluation & Latent Space Analysis")
    
    # KPI Metric Cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-val">{metrics['accuracy']*100:.2f}%</div>
            <div class="stat-lbl">Test Set Accuracy</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        covid_recall = metrics['report'].get('covid', {}).get('recall', 0.0)
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-val">{covid_recall*100:.2f}%</div>
            <div class="stat-lbl">COVID-19 Sensitivity / Recall</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        covid_prec = metrics['report'].get('covid', {}).get('precision', 0.0)
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-val">{covid_prec*100:.2f}%</div>
            <div class="stat-lbl">COVID-19 Precision</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        total_var = metrics.get('total_variance', 0.0)
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-val">{total_var:.1f}%</div>
            <div class="stat-lbl">PCA Retained Variance (100 PCs)</div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    col_cm, col_pca_plot = st.columns([1, 1], gap="large")

    with col_cm:
        st.markdown("#### Test Confusion Matrix")
        cm_data = metrics['confusion_matrix']
        cm_df = pd.DataFrame(
            cm_data,
            index=["Actual COVID", "Actual Normal"],
            columns=["Predicted COVID", "Predicted Normal"]
        )
        fig_cm = px.imshow(
            cm_df,
            text_auto=True,
            color_continuous_scale="Blues",
            labels=dict(x="Predicted", y="Actual", color="Count")
        )
        fig_cm.update_layout(
            height=360,
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color="#cbd5e1")
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    with col_pca_plot:
        st.markdown("#### Latent Feature Space (PC1 vs PC2 vs PC3)")
        test_samples = pipeline.get('test_pca_samples', {})
        if test_samples:
            scatter_df = pd.DataFrame({
                "PC 1": test_samples['pc1'],
                "PC 2": test_samples['pc2'],
                "PC 3": test_samples['pc3'],
                "Class": ["COVID-19" if t == 'covid' else "Normal" for t in test_samples['true_label']]
            })
            fig_3d = px.scatter_3d(
                scatter_df,
                x="PC 1",
                y="PC 2",
                z="PC 3",
                color="Class",
                color_discrete_map={"COVID-19": "#ef4444", "Normal": "#10b981"},
                opacity=0.75,
                title="Separation of COVID vs Normal Radiographs in PCA Latent Space"
            )
            fig_3d.update_layout(
                height=380,
                paper_bgcolor='rgba(0,0,0,0)',
                scene=dict(
                    xaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="#334155"),
                    yaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="#334155"),
                    zaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="#334155")
                ),
                font=dict(color="#cbd5e1"),
                margin=dict(l=0, r=0, t=30, b=0)
            )
            st.plotly_chart(fig_3d, use_container_width=True)

    st.divider()
    st.markdown("#### Cumulative Explained Variance Ratio across Principal Components")
    var_ratios = np.cumsum(metrics['explained_variance_ratio']) * 100
    var_df = pd.DataFrame({
        "Component": list(range(1, len(var_ratios) + 1)),
        "Cumulative Explained Variance (%)": var_ratios
    })
    fig_var = px.line(
        var_df,
        x="Component",
        y="Cumulative Explained Variance (%)",
        markers=True
    )
    fig_var.update_layout(
        height=280,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color="#cbd5e1")
    )
    st.plotly_chart(fig_var, use_container_width=True)

# ---------------------------------------------------------
# TAB 3: SYSTEM & ARCHITECTURE
# ---------------------------------------------------------
else:
    st.subheader("System Architecture & Diagnostic Methodology")
    st.markdown("""
    ### End-to-End Diagnostic Pipeline
    ```mermaid
    graph LR
        A[Chest Radiograph Image] --> B[64x64 Grayscale Flattening 4096 px]
        B --> C[StandardScaler Normalization]
        C --> D[PCA Dimension Reduction 100 Components]
        D --> E[BaggingClassifier: 50x SVC RBF Kernel]
        E --> F[Probability & Decision Output]
    ```
    """)

    st.markdown("""
    ### Why this architecture?
    1. **Principal Component Analysis (PCA)**:
       - Compresses high-dimensional 4,096 pixel intensity arrays into 100 orthogonal eigenvectors.
       - Filters high-frequency spatial noise and acquisition artifacts across different X-ray machines.
       - Accelerates training and inference from minutes to milliseconds.

    2. **Bagging Ensemble with Support Vector Machines**:
       - Support Vector Machines with Radial Basis Function (RBF) kernels construct non-linear hyperplanes in the 100-dimensional eigenspace.
       - Bootstrap aggregating (Bagging) across 50 distinct sub-estimators prevents overfitting and variance spikes.
       - Calibrated probability estimates enable transparent clinical confidence scoring.
    """)
