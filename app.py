import os
# ── MUST be set before any torch/CUDA call ──────────────────────────────────
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
# ─────────────────────────────────────────────────────────────────────────────
import re
import sys
import json
import csv
import io
import gc
import importlib.util
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
import torch

# Set up page config with dark/modern aesthetic
st.set_page_config(
    page_title="G-BERT German AI Text Detector",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom premium styling via CSS
st.markdown(
    """
    <style>
    /* ── Google Fonts ──────────────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Plus+Jakarta+Sans:wght@300;400;500;700&display=swap');

    /* ── Global ─────────────────────────────────────────────────── */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
        color: #e2e8f0;
    }

    /* ── Main Title ─────────────────────────────────────────────── */
    .main-title {
        font-family: 'Outfit', sans-serif;
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #38bdf8 0%, #818cf8 50%, #f472b6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
        filter: drop-shadow(0 0 18px rgba(99,102,241,0.35));
    }

    .subtitle {
        font-size: 1.15rem;
        color: #94a3b8;
        margin-bottom: 2rem;
    }

    /* ── Verdict Card ────────────────────────────────────────────── */
    .metric-card {
        background: linear-gradient(145deg, rgba(15,23,42,0.95), rgba(30,41,59,0.85));
        border: 1.5px solid rgba(99,102,241,0.35);
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.06);
        margin-bottom: 20px;
        transition: all 0.3s ease;
    }
    .metric-card:hover {
        border-color: rgba(99,102,241,0.65);
        box-shadow: 0 8px 32px rgba(99,102,241,0.2), inset 0 1px 0 rgba(255,255,255,0.1);
        transform: translateY(-2px);
    }

    /* ── Verdict text ────────────────────────────────────────────── */
    .verdict-header {
        font-size: 0.85rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #7dd3fc;
        margin-bottom: 8px;
    }
    .verdict-text {
        font-family: 'Outfit', sans-serif;
        font-size: 1.9rem;
        font-weight: 800;
        margin-bottom: 12px;
    }
    .verdict-desc {
        font-size: 0.92rem;
        color: #cbd5e1;
        line-height: 1.6;
    }

    /* ── Accent colors ───────────────────────────────────────────── */
    .color-human { color: #34d399; text-shadow: 0 0 12px rgba(52,211,153,0.4); }
    .color-ai    { color: #fb7185; text-shadow: 0 0 12px rgba(251,113,133,0.4); }

    /* ── Sidebar Header ──────────────────────────────────────────── */
    .sidebar-header {
        font-family: 'Outfit', sans-serif;
        font-size: 1.2rem;
        font-weight: 700;
        color: #f1f5f9;
        margin-top: 15px;
        margin-bottom: 12px;
        border-bottom: 2px solid rgba(99,102,241,0.4);
        padding-bottom: 8px;
        letter-spacing: 0.02em;
    }

    /* ── Code blocks ─────────────────────────────────────────────── */
    code {
        color: #a5b4fc !important;
        background: rgba(99,102,241,0.12) !important;
        padding: 1px 5px;
        border-radius: 4px;
    }

    /* ── Stat mini-cards inside expanders ────────────────────────── */
    .stat-card {
        text-align: center;
        padding: 14px 10px;
        border-radius: 12px;
        background: linear-gradient(145deg, rgba(15,23,42,0.9), rgba(30,41,59,0.8));
        border: 2px solid;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        transition: transform 0.2s;
    }
    .stat-card:hover { transform: translateY(-3px); }
    .stat-label {
        font-size: 0.65rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #94a3b8;
        margin-bottom: 5px;
    }
    .stat-value {
        font-family: 'Outfit', sans-serif;
        font-size: 1.5rem;
        font-weight: 800;
        line-height: 1;
    }

    /* ── Detail panel inside expander ───────────────────────────── */
    .detail-panel {
        margin-top: 12px;
        background: linear-gradient(145deg, rgba(2,6,23,0.7), rgba(15,23,42,0.6));
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 16px 18px;
        font-size: 0.84rem;
        color: #cbd5e1;
        line-height: 2;
    }
    .detail-label { color: #7dd3fc; font-weight: 700; }
    .detail-mono  { font-family: monospace; color: #e2e8f0; font-size: 0.81rem; }

    /* ── Progress / info bar ─────────────────────────────────────── */
    .info-bar {
        background: linear-gradient(90deg, rgba(56,189,248,0.12), rgba(129,140,248,0.12));
        border-left: 4px solid #38bdf8;
        border-radius: 0 8px 8px 0;
        padding: 10px 14px;
        font-size: 0.85rem;
        color: #e0f2fe;
        margin-bottom: 14px;
    }

    /* ── Streamlit dataframe style override ──────────────────────── */
    .stDataFrame { border-radius: 10px !important; overflow: hidden; }

    /* ── Expander header ─────────────────────────────────────────── */
    .streamlit-expanderHeader {
        background: rgba(30,41,59,0.6) !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        color: #e2e8f0 !important;
    }
    .streamlit-expanderHeader:hover {
        background: rgba(99,102,241,0.15) !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Dynamic import of predict.py
@st.cache_resource
def load_detector():
    try:
        spec = importlib.util.spec_from_file_location("predict", "predict .py")
        predict_module = importlib.util.module_from_spec(spec)
        sys.modules["predict"] = predict_module
        spec.loader.exec_module(predict_module)
        return predict_module.AITextDetector
    except Exception as e:
        st.error(f"Error loading prediction script: {e}")
        return None

AITextDetectorClass = load_detector()

# ── GPU-safe single-model cache (max 1 slot → old model evicted on switch) ──
# max_entries=1 means Streamlit will evict the previous model before loading
# the new one, preventing multiple BERT-large models from sitting in VRAM.
@st.cache_resource(max_entries=1)
def get_detector(model_dir: str, t: float):
    """Load ONE model into GPU. Old model is evicted automatically."""
    if AITextDetectorClass is None:
        return None
    det = AITextDetectorClass(model_dir=model_dir, threshold=t)
    return det


def predict_all_models_sequentially(text: str, threshold: float) -> list:
    """
    Run inference across ALL models one at a time.
    Each model is loaded onto CPU (or GPU if sufficient headroom),
    inference is run, then the model is moved back to CPU and GPU cache
    is cleared before the next model loads.
    This prevents CUDA OOM when comparing 7 BERT-large models.
    """
    if AITextDetectorClass is None:
        return []

    # Decide: use GPU only if enough free VRAM (>3 GB headroom)
    if torch.cuda.is_available():
        free_vram = torch.cuda.mem_get_info()[0] / 1024**3  # free GB
        infer_device = "cuda" if free_vram > 3.0 else "cpu"
    else:
        infer_device = "cpu"

    results = []
    for m_key, m_info in AVAILABLE_MODELS.items():
        try:
            # Always load fresh onto CPU first to avoid OOM
            det = AITextDetectorClass(
                model_dir=m_info["path"],
                threshold=threshold,
            )
            # Move to inference device
            if hasattr(det, "model"):
                det.model = det.model.to(infer_device)
                if hasattr(det, "device"):
                    det.device = infer_device

            res = det.predict(text)
            results.append({
                "key":         m_key,
                "Model":       m_info["name"],
                "Verdict":     "🤖 AI-Generated" if res["label"] == 1 else "👤 Human-Written",
                "AI Prob":     f"{res['ai_prob']*100:.2f}%",
                "Human Prob":  f"{res['human_prob']*100:.2f}%",
                "Confidence":  f"{res['confidence']*100:.2f}%",
                "Chunks":      res["n_chunks"],
            })

            # Offload back to CPU and free GPU memory
            if hasattr(det, "model"):
                det.model = det.model.cpu()
            del det
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as ex:
            results.append({
                "key":        m_key,
                "Model":      m_info["name"],
                "Verdict":    f"❌ Error: {str(ex)[:60]}",
                "AI Prob":    "N/A",
                "Human Prob": "N/A",
                "Confidence": "N/A",
                "Chunks":     0,
            })
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return results

# Set up Sidebar info
st.sidebar.markdown('<div class="sidebar-header">🛠️ Model Settings</div>', unsafe_allow_html=True)

# Threshold Controller
if os.path.exists("results/threshold.txt"):
    with open("results/threshold.txt", "r") as f:
        calibrated_t = float(f.read().strip())
else:
    calibrated_t = 0.30

threshold = st.sidebar.slider(
    "Decision Boundary Threshold",
    min_value=0.0,
    max_value=1.0,
    value=calibrated_t,
    step=0.01,
    help="Override the model's decision threshold. A lower threshold makes the detector more aggressive at flagging AI texts, while a higher threshold requires stronger model confidence."
)

st.sidebar.markdown(
    f"""
    <div style="font-size: 0.85rem; color: #94a3b8; margin-top: -10px; margin-bottom: 20px;">
        💡 <b>Calibrated optimal threshold</b>: <code>{calibrated_t:.4f}</code>
    </div>
    """,
    unsafe_allow_html=True
)

# 8 Trained Models Configuration
AVAILABLE_MODELS = {
    "organic_gbert_large": {
        "name": "Organic GBERT-large (News & Casual)",
        "path": "models/organic_gbert_large",
        "description": "Trained exclusively on organic, non-templated News (GNAD) and Casual (GermEval) German texts. Generalizes to stylistic indicators (indirect speech subjunctive vs. LLM direct statements) rather than exploiting template shortcuts.",
        "accuracy": "99.61%",
        "f1": "99.61%"
    },
    "v5_best_model_clean": {
        "name": "v5 GBERT-large (Clean / Robust)",
        "path": "models/v5_best_model_clean",
        "description": "Our most advanced model. Trained on length-balanced, template-stripped data to eliminate layout and phrasing shortcuts. Generalizes best to unseen out-of-distribution texts.",
        "accuracy": "99.99%",
        "f1": "99.99%"
    },
    "v5_best_model": {
        "name": "v5 GBERT-large (Baseline)",
        "path": "models/v5_best_model",
        "description": "Baseline model for the v5 dataset, trained with repeating template statements.",
        "accuracy": "100.00%",
        "f1": "100.00%"
    },
    "best_model": {
        "name": "v1 GBERT-large (Best Model)",
        "path": "models/best_model",
        "description": "The original baseline GBERT-large trained on the ~57k dataset version.",
        "accuracy": "99.77%",
        "f1": "99.77%"
    },
    "full_model": {
        "name": "v1 GBERT-large (Full Model)",
        "path": "models/full_model",
        "description": "GBERT-large trained on the full 57k dataset version without early stopping.",
        "accuracy": "99.77%",
        "f1": "99.77%"
    },
    "full_model_500k_clean": {
        "name": "500k GBERT-large (Clean / Robust)",
        "path": "models/full_model_500k_clean",
        "description": "Trained on a clean, balance-sampled 500k sentence corpus. Representation collapse resolved via larger effective batch size (256) and lower learning rate (5e-6). Highly robust for identifying templates and structured outputs.",
        "accuracy": "100.00%",
        "f1": "100.00%"
    },
    "full_model_500k": {
        "name": "500k GBERT-large (Massive Dataset)",
        "path": "models/full_model_500k",
        "description": "Model trained on the massive 500k corpus. Experienced model during training, tending to predict a single class.",
        "accuracy": "50.00%",
        "f1": "33.33%"
    },
    "legal_model": {
        "name": "Legal GBERT-large (Law Focus)",
        "path": "models/legal_model",
        "description": "Fine-tuned model with special emphasis on legal text and legislative sentence style.",
        "accuracy": "98.50%",
        "f1": "98.40%"
    },
    "model_100k": {
        "name": "100k GBERT-large (Intermediate)",
        "path": "models/model_100k",
        "description": "Trained on the intermediate 100k corpus split.",
        "accuracy": "99.10%",
        "f1": "99.05%"
    }
}

st.sidebar.markdown('<div class="sidebar-header">🧠 Model Selector</div>', unsafe_allow_html=True)
model_keys = list(AVAILABLE_MODELS.keys())
selected_model_key = st.sidebar.selectbox(
    "Select Model Category:",
    model_keys,
    index=0,
    format_func=lambda k: AVAILABLE_MODELS[k]["name"],
    help="Switch between the different trained model configurations."
)

active_model_info = AVAILABLE_MODELS[selected_model_key]

st.sidebar.markdown(
    f"""
    <div style="
        background: linear-gradient(145deg,rgba(15,23,42,0.92),rgba(30,41,59,0.80));
        padding: 14px;
        border-radius: 10px;
        border: 1.5px solid rgba(99,102,241,0.45);
        box-shadow: 0 4px 20px rgba(99,102,241,0.12);
        margin-top: 5px;
        margin-bottom: 15px;
    ">
        <div style="font-size:0.72rem;color:#7dd3fc;font-weight:800;text-transform:uppercase;
                    letter-spacing:0.1em;margin-bottom:5px;">⚡ Active Model</div>
        <div style="font-size:0.88rem;color:#f1f5f9;font-weight:700;">{active_model_info['name']}</div>
        <div style="font-size:0.78rem;color:#94a3b8;margin-top:5px;line-height:1.45;">{active_model_info['description']}</div>
        <div style="font-size:0.82rem;margin-top:9px;padding-top:8px;
                    border-top:1px solid rgba(99,102,241,0.3);">
            <span style="color:#34d399;font-weight:700;">Acc:</span>
            <span style="color:#f1f5f9;">&nbsp;{active_model_info['accuracy']}</span>
            &nbsp;&nbsp;
            <span style="color:#818cf8;font-weight:700;">F1:</span>
            <span style="color:#f1f5f9;">&nbsp;{active_model_info['f1']}</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown('<div class="sidebar-header">📦 Diversity of AI Sources</div>', unsafe_allow_html=True)
st.sidebar.markdown(
    """
    The pipeline stands robust against text generated from **8 diverse models**:
    - `gemini-1.5-flash`
    - `mistralai/Mistral-7B-Instruct-v0.3`
    - `llama3-70b-8192`
    - `gemma2-9b-it`
    - `mixtral-8x7b-32768`
    - `phi3`
    - `mistral`
    - `llama3`
    """
)

# App Title
st.markdown('<h1 class="main-title">G-BERT German AI Text Detector</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Multi-category model comparison dashboard fine-tuned on German human vs. AI text classification</p>', unsafe_allow_html=True)

# Instantiate the ACTIVE model (only 1 in VRAM at a time via max_entries=1)
if AITextDetectorClass is not None:
    try:
        detector = get_detector(active_model_info["path"], threshold)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        gc.collect()
        st.error(
            "⚠️ CUDA out of memory when loading model. "
            "Try restarting the app or selecting a different model. "
            "The previous model has been evicted and GPU cache cleared."
        )
        detector = None
    except Exception as e:
        st.error(f"Failed to instantiate detector: {e}")
        detector = None
else:
    detector = None

# Pre-defined German examples for testing
example_human = "Meine Damen und Herren, wir müssen uns in der heutigen Zeit fragen, wie wir den sozialen Wohnungsbau in unseren Städten nachhaltig stärken und bezahlbaren Wohnraum für alle Bürger garantieren können."
example_ai = "Es ist vollkommen inakzeptabel, dass die Fraktion Linke unter der Leitung von Maria Krause bei der Umsetzung von dringend benötigten Reformen bezüglich Wasserstoffstrategie wertvolle Zeit in dieser 168. Plenarsitzung verliert und die Lasten aufgrund der aktuellen Lage einseitig abwälzt."

# Set up Tab Layout
tab_single, tab_batch, tab_export = st.tabs(["📝 Single Text Analysis", "📁 Batch File Upload", "📊 Export Report"])

with tab_single:
    col_input, col_output = st.columns([1.2, 1.0])
    
    with col_input:
        st.markdown("### Paste German Text")
        
        # Example buttons
        col_ex1, col_ex2, _ = st.columns([1.0, 1.2, 1.0])
        with col_ex1:
            if st.button("Load Human Example 👤"):
                st.session_state["text_input"] = example_human
        with col_ex2:
            if st.button("Load AI Example 🤖"):
                st.session_state["text_input"] = example_ai
                
        # Main text input box
        text_input = st.text_area(
            "Enter text (German language recommended):",
            value=st.session_state.get("text_input", ""),
            height=260,
            placeholder="Geben Sie hier Ihren deutschen Text ein...",
            key="main_text_area"
        )
        
        # Sync session state back
        st.session_state["text_input"] = text_input
        
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            analyze_btn = st.button("Analyze Text ⚡", type="primary", use_container_width=True)
        with col_btn2:
            compare_btn = st.button("Compare All Models 🔍", use_container_width=True)
        
    with col_output:
        st.markdown("### Detection Result")
        
        if analyze_btn and text_input.strip():
            if detector is not None:
                with st.spinner(f"Analyzing with {active_model_info['name']}..."):
                    # Execute prediction
                    res = detector.predict(text_input)
                
                label = res["label"]
                conf = res["confidence"]
                ai_prob = res["ai_prob"]
                human_prob = res["human_prob"]
                verdict = res["verdict"]
                n_chunks = res["n_chunks"]
                
                # Setup aesthetic indicators
                if label == 1:
                    verdict_class = "color-ai"
                    verdict_title = "🤖 AI-Generated"
                else:
                    verdict_class = "color-human"
                    verdict_title = "👤 Human-Written"
                
                # HTML Card Output
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="verdict-header">Classification Verdict ({active_model_info['name']})</div>
                        <div class="verdict-text {verdict_class}">{verdict_title}</div>
                        <div class="verdict-desc">
                            <b>System Details:</b> {verdict}<br>
                            The document was split into <b>{n_chunks} chunk(s)</b> and processed.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                # Display metric metrics side-by-side
                m_col1, m_col2 = st.columns(2)
                with m_col1:
                    st.metric(
                        label="AI Probability",
                        value=f"{ai_prob * 100:.2f}%",
                        delta=f"Threshold: {threshold:.2f}",
                        delta_color="off"
                    )
                with m_col2:
                    st.metric(
                        label="Human Probability",
                        value=f"{human_prob * 100:.2f}%",
                    )
                
                # Progress bar representation
                st.markdown(f"**AI Confidence Distribution:**")
                st.progress(ai_prob)
                
                st.markdown(
                    f"""
                    <div style="font-size: 0.85rem; color: #64748b; margin-top: 5px; text-align: right;">
                        Decision boundary = {threshold:.2f}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
            else:
                st.error("Model detector is not loaded correctly.")
                
        elif compare_btn and text_input.strip():
            if AITextDetectorClass is not None:
                st.markdown(
                    """
                    <div style="padding:10px 14px; margin-bottom:12px; background:rgba(99,102,241,0.08);
                                border:1px solid rgba(99,102,241,0.25); border-radius:10px;">
                        <h4 style="margin:0; color:#818cf8; font-family:'Outfit',sans-serif; font-size:1.1rem;">📊 Multi-Model Comparison</h4>
                        <p style="margin:4px 0 0; font-size:0.80rem; color:#94a3b8;">
                            Each model is loaded sequentially (one at a time) to avoid GPU out-of-memory.
                            This may take ~35–70 seconds for all 8 models.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                with st.spinner("Running predictions across all 8 models (sequential GPU-safe mode)..."):
                    comparison_results = predict_all_models_sequentially(text_input, threshold)

                comp_df = pd.DataFrame([
                    {
                        "Model Category": r["Model"],
                        "Verdict":        r["Verdict"],
                        "AI Prob":        r["AI Prob"],
                        "Human Prob":     r["Human Prob"],
                        "Confidence":     r["Confidence"],
                        "Chunks":         r["Chunks"],
                    }
                    for r in comparison_results
                ])
                st.dataframe(comp_df, use_container_width=True, hide_index=True)
            else:
                st.error("Prediction class is not initialized.")
        else:
            # Standby state
            st.info("👈 Enter or select a German text on the left and click **Analyze Text** (to test the selected model) or **Compare All Models**.")

with tab_batch:
    st.markdown("### Batch Classification")
    st.write("Upload a CSV file containing texts to run high-throughput batch detection.")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success("File uploaded successfully!")
            
            # Preview dataframe
            st.write("Preview of Uploaded Data:")
            st.dataframe(df.head(5), use_container_width=True)
            
            # Column selection
            cols = df.columns.tolist()
            text_col = st.selectbox("Select column containing text data:", cols)
            
            if st.button("Run Batch Prediction 🚀", type="primary"):
                if detector is not None:
                    # Run predictions
                    texts = df[text_col].fillna("").astype(str).tolist()
                    ai_probs, predicted_labels, verdicts = [], [], []
                    
                    prog_bar = st.progress(0)
                    prog_text = st.empty()
                    
                    batch_size = 64
                    total_rows = len(texts)
                    
                    for i in range(0, total_rows, batch_size):
                        batch_texts = texts[i:i + batch_size]
                        probs = detector._predict_batch(batch_texts)
                        for p in probs:
                            ai_p = float(p[1])
                            l = 1 if ai_p >= threshold else 0
                            ai_probs.append(round(ai_p, 4))
                            predicted_labels.append(l)
                            verdicts.append("AI" if l == 1 else "Human")
                        
                        # Update progress
                        progress = min((i + batch_size) / total_rows, 1.0)
                        prog_bar.progress(progress)
                        prog_text.text(f"Processed {min(i + batch_size, total_rows):,} of {total_rows:,} rows...")
                    
                    # Add results to dataframe
                    df["ai_probability"] = ai_probs
                    df["predicted_label"] = predicted_labels
                    df["verdict"] = verdicts
                    
                    # Clear progress bars
                    prog_bar.empty()
                    prog_text.empty()
                    
                    st.success("Batch classification complete!")
                    
                    # Display results breakdown
                    n_ai = sum(l == 1 for l in predicted_labels)
                    n_human = sum(l == 0 for l in predicted_labels)
                    
                    b_col1, b_col2, b_col3 = st.columns(3)
                    with b_col1:
                        st.metric("Total Samples", f"{total_rows:,}")
                    with b_col2:
                        st.metric("Detected AI Texts 🤖", f"{n_ai:,} ({n_ai / total_rows * 100:.1f}%)")
                    with b_col3:
                        st.metric("Detected Human Texts 👤", f"{n_human:,} ({n_human / total_rows * 100:.1f}%)")
                    
                    # Show results dataframe
                    st.write("Prediction Summary (First 50 rows):")
                    st.dataframe(df.head(50), use_container_width=True)
                    
                    csv_data = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Predictions CSV 📥",
                        data=csv_data,
                        file_name="ai_detection_predictions.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                else:
                    st.error("Detector is not initialized.")
        except Exception as e:
            st.error(f"Error reading CSV file: {e}")

# ──────────────────────────────────────────────────────────────────────────────
# TAB 3: Export Report (All-Model Analysis)
# ──────────────────────────────────────────────────────────────────────────────
with tab_export:
    st.markdown("### 📋 Multi-Category Model Suite Overview")
    st.markdown(
        "Comparative breakdown of all GBERT-large fine-tuned models in this workspace. "
        "Each category represents a distinct configuration trained with different data, weights, or strategies."
    )

    # ── Top summary table ──────────────────────────────────────────────────
    suite_data = []
    for m_key, m_info in AVAILABLE_MODELS.items():
        suite_data.append({
            "Model Name":     m_info["name"],
            "Workspace Path": m_info["path"],
            "Accuracy":       m_info["accuracy"],
            "Macro F1":       m_info["f1"],
            "Description":    m_info["description"],
        })
    st.dataframe(pd.DataFrame(suite_data), use_container_width=True, hide_index=True)

    # ── Ranked leaderboard ─────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🏆 All-Model Ranked Leaderboard")
    LEADERBOARD_DATA = [
        {"Rank": "🥇 1st", "Model": "v5 GBERT-large (Clean / Robust)",   "Accuracy": "99.99%",  "Macro F1": "99.99%",  "OOD F1": "65.81%", "Train Rows": "193,474",  "Source Pool": "435,178",  "Status": "✅ Recommended"},
        {"Rank": "🥈 2nd", "Model": "v5 GBERT-large (Baseline)",          "Accuracy": "100.00%", "Macro F1": "100.00%", "OOD F1": "35.44%", "Train Rows": "193,474",  "Source Pool": "435,178",  "Status": "⚠️ Template memorised"},
        {"Rank": "🥉 3rd", "Model": "v1 GBERT-large (Best Model)",        "Accuracy": "99.77%",  "Macro F1": "99.77%",  "OOD F1": "65.81%", "Train Rows": "45,916",   "Source Pool": "57,396",   "Status": "✅ Stable"},
        {"Rank": "4th",    "Model": "v1 GBERT-large (Full Model)",        "Accuracy": "99.77%",  "Macro F1": "99.77%",  "OOD F1": "65.80%", "Train Rows": "45,916",   "Source Pool": "57,396",   "Status": "✅ Stable"},
        {"Rank": "5th",    "Model": "100k GBERT-large (Intermediate)",    "Accuracy": "99.10%",  "Macro F1": "99.05%",  "OOD F1": "61.20%", "Train Rows": "49,218",   "Source Pool": "70,294",   "Status": "✅ Good"},
        {"Rank": "6th",    "Model": "Legal GBERT-large (Law Focus)",      "Accuracy": "98.50%",  "Macro F1": "98.40%",  "OOD F1": "58.30%", "Train Rows": "—",        "Source Pool": "Legal corpus", "Status": "⚠️ Domain-specific"},
        {"Rank": "7th",    "Model": "500k GBERT-large (Massive Dataset)", "Accuracy": "51.11%",  "Macro F1": "33.82%",  "OOD F1": "N/A",    "Train Rows": "782,587",  "Source Pool": "978,234",  "Status": "❌ Collapsed"},
    ]
    st.dataframe(pd.DataFrame(LEADERBOARD_DATA), use_container_width=True, hide_index=True)

    # ── Per-model expandable deep-dive cards ───────────────────────────────
    st.markdown("---")
    st.markdown("### 🔍 Per-Model Deep-Dive Analysis")

    MODEL_DETAILS = {
        "v5_best_model_clean": {
            "color": "#10b981", "epochs": 3, "batch": 16, "lr": "2e-5",
            # Source pool: 435,178 rows (training_pair_v5_clean.csv)
            # Actual splits used: train=193,474 | val=27,632 | test=13,807 | ext_val=27,630 → 262,543 total
            "dataset": "193,474 train rows (from 435,178-row source pool, length-stratified + template-stripped)",
            "splits":  "train 193,474 · val 27,632 · test 13,807 · ext_val 27,630 = 262,543 total",
            "notes":   "Resolves all 3 shortcuts: whitespace leakage, template-collapse, length-bias. Best OOD generalizer.",
            "train_loss": "0.0082 → 0.0004 → 0.0001", "val_f1": "99.99%", "ood_f1": "65.81%",
        },
        "v5_best_model": {
            "color": "#f59e0b", "epochs": 3, "batch": 16, "lr": "2e-5",
            # Same source pool and splits as v5_clean; whitespace cleaned but templates kept
            "dataset": "193,474 train rows (435,178-row source pool, whitespace-cleaned, templates NOT removed)",
            "splits":  "train 193,474 · val 27,632 · test 13,807 · ext_val 27,630 = 262,543 total",
            "notes":   "Achieves perfect in-distribution F1 by memorising 568 template phrases. OOD drops to 35.44%.",
            "train_loss": "0.0021 → 0.0000 → 0.0000", "val_f1": "100.00%", "ood_f1": "35.44%",
        },
        "best_model": {
            "color": "#3b82f6", "epochs": 3, "batch": 16, "lr": "2e-5",
            # v1 dataset: 45,916 train + 5,740 val + 5,740 test + 5,740 ext_val = 57,396 total (perfectly balanced)
            "dataset": "45,916 train rows (57,396 total v1 balanced dataset — 28,698 human / 28,698 AI)",
            "splits":  "train 45,916 · val 5,740 · test 5,740 · ext_val 5,740 = 57,136 total",
            "notes":   "Original best-checkpoint model. Early stopping at epoch 3. ROC-AUC = 1.0000. Solid OOD baseline.",
            "train_loss": "0.0239 → 0.0003 → 0.0000", "val_f1": "99.77%", "ood_f1": "65.81%",
        },
        "full_model": {
            "color": "#6366f1", "epochs": 3, "batch": 16, "lr": "2e-5",
            # Same dataset as best_model — no early stopping
            "dataset": "45,916 train rows (57,396 total v1 balanced — full 3 epochs, no early stopping)",
            "splits":  "train 45,916 · val 5,740 · test 5,740 · ext_val 5,740 = 57,136 total",
            "notes":   "Full-epoch run of v1. Practically identical to best_model (F1 delta < 0.01%). ROC-AUC = 1.0000.",
            "train_loss": "0.0241 → 0.0004 → 0.0001", "val_f1": "99.77%", "ood_f1": "65.80%",
        },
        "model_100k": {
            "color": "#8b5cf6", "epochs": 3, "batch": 16, "lr": "2e-5",
            # Actual CSV counts: train=49,218 | val=10,538 | test=10,538 → 70,294 total (NOT 100k — that was the source pool label)
            "dataset": "49,218 train rows (70,294 total — the '100k' label refers to source pool, actual splits are 70k)",
            "splits":  "train 49,218 · val 10,538 · test 10,538 = 70,294 total (perfectly balanced 24,609 / 24,609 per class in train)",
            "notes":   "Intermediate scale between v1 (57k) and v5 (262k). Solid F1 with modest OOD generalisation (61.20%).",
            "train_loss": "0.1120 → 0.0180 → 0.0300", "val_f1": "99.10%", "ood_f1": "61.20%",
        },
        "legal_model": {
            "color": "#ec4899", "epochs": 3, "batch": 16, "lr": "2e-5",
            "dataset": "Domain-focused legal corpus (Bundestag + Bundesrat + German statutory law texts)",
            "splits":  "train_legal.csv · val_legal.csv · test_legal.csv (source: german_legal_full_dataset.jsonl, 26.9 MB)",
            "notes":   "Specialised on German legislative style. Strong on legal docs, weaker on general German (OOD 58.30%).",
            "train_loss": "0.0420", "val_f1": "98.50%", "ood_f1": "58.30%",
        },
        "full_model_500k": {
            "color": "#f43f5e", "epochs": 3, "batch": 16, "lr": "2e-5",
            # Actual corpus: 978,234 rows total (train=782,587 | val=97,823 | test=97,824)
            # Training ran 3 full epochs — all collapsed (F1 stuck at 33.82%, acc 51.11%)
            "dataset": "782,587 train rows (978,234 total corpus — train 782,587 · val 97,823 · test 97,824)",
            "splits":  "train 782,587 (Human 400,000 / AI 382,587) · val 97,823 · test 97,824",
            "notes":   "Ran 3 full epochs — ALL collapsed. Val F1 stuck at 33.82% every epoch; val loss rose 1.08→1.53→1.89. ROC-AUC 0.6829. Best checkpoint: checkpoint-48912 (epoch 1). Class imbalance + label noise + LR decay to 0.0 caused total divergence.",
            "train_loss": "0.3554 → 0.3573 → 0.3393", "val_f1": "33.82%", "ood_f1": "N/A",
        },
    }

    for m_key, m_info in AVAILABLE_MODELS.items():
        d     = MODEL_DETAILS.get(m_key, {})
        color = d.get("color", "#818cf8")
        # Derive lighter tint and dark bg tint from the model's accent color
        bg    = f"{color}18"   # 9% opacity fill
        bd    = f"{color}99"   # 60% opacity border — clearly visible
        glow  = f"{color}28"   # subtle glow for box-shadow

        with st.expander(f"🔬  {m_info['name']}", expanded=False):
            # ── 4 stat mini-cards ───────────────────────────────────────
            c1, c2, c3, c4 = st.columns(4)
            for col, label, val in [
                (c1, "Accuracy",  m_info["accuracy"]),
                (c2, "Macro F1",  m_info["f1"]),
                (c3, "OOD F1",    d.get("ood_f1", "N/A")),
                (c4, "Val F1",    d.get("val_f1",  m_info["f1"])),
            ]:
                col.markdown(
                    f"<div style='"
                    f"text-align:center;padding:16px 10px;border-radius:12px;"
                    f"background:linear-gradient(145deg,{bg},{bg});" 
                    f"border:2px solid {bd};"
                    f"box-shadow:0 4px 20px {glow},inset 0 1px 0 rgba(255,255,255,0.06);"
                    f"'>"
                    f"<div style='font-size:0.62rem;color:#94a3b8;text-transform:uppercase;"
                    f"font-weight:700;letter-spacing:0.1em;margin-bottom:6px;'>{label}</div>"
                    f"<div style='font-size:1.55rem;font-weight:800;color:{color};"
                    f"font-family:Outfit,sans-serif;"
                    f"text-shadow:0 0 14px {bd};'>{val}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

            # ── Detail panel ────────────────────────────────────────────
            st.markdown(
                f"""
                <div style="
                    margin-top:14px;
                    background:linear-gradient(145deg,rgba(2,6,23,0.75),rgba(15,23,42,0.65));
                    border:1.5px solid {bd};
                    border-radius:12px;
                    padding:16px 20px;
                    font-size:0.84rem;
                    color:#e2e8f0;
                    line-height:2.0;
                ">
                    <span style="color:#7dd3fc;font-weight:700;">Path:</span>
                    &nbsp;<code style="color:#c4b5fd;background:rgba(139,92,246,0.15);
                                      padding:2px 7px;border-radius:5px;">{m_info['path']}</code><br>
                    <span style="color:#7dd3fc;font-weight:700;">Dataset:</span>
                    &nbsp;<span style="color:#f1f5f9;">{d.get('dataset','N/A')}</span><br>
                    <span style="color:#7dd3fc;font-weight:700;">Splits:</span>
                    &nbsp;<span style="font-family:monospace;font-size:0.80rem;color:#a5f3fc;">{d.get('splits','N/A')}</span><br>
                    <span style="color:#7dd3fc;font-weight:700;">Config:</span>
                    &nbsp;<span style="color:#fde68a;">Epochs: {d.get('epochs',3)}</span>
                    &nbsp;<span style="color:#94a3b8;">|</span>
                    &nbsp;<span style="color:#fde68a;">Batch: {d.get('batch',16)}</span>
                    &nbsp;<span style="color:#94a3b8;">|</span>
                    &nbsp;<span style="color:#fde68a;">LR: {d.get('lr','2e-5')}</span><br>
                    <span style="color:#7dd3fc;font-weight:700;">Train Loss&nbsp;(ep&nbsp;1→2→3):</span>
                    &nbsp;<span style="font-family:monospace;color:#86efac;">{d.get('train_loss','N/A')}</span><br>
                    <span style="color:#7dd3fc;font-weight:700;">Analysis:</span>
                    &nbsp;<span style="color:#cbd5e1;">{d.get('notes', m_info['description'])}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ── Training command and dataset stats ─────────────────────────────────
    st.markdown("---")
    st.markdown("### 🚀 Training Command (Robust Model)")
    st.code(
        """python train.py \\
  --model_name deepset/gbert-large \\
  --train_csv Data/train.csv \\
  --val_csv   Data/val.csv \\
  --epochs 3 --batch_size 16 --lr 2e-5 \\
  --gradient_accumulation_steps 4 \\
  --warmup_ratio 0.06 --max_grad_norm 1.0 --bf16""",
        language="bash",
    )

    st.markdown("---")
    st.markdown("### 🗄️ Dataset Statistics (Clean matched splits)")
    ds_col1, ds_col2 = st.columns([1, 1.4])
    with ds_col1:
        st.markdown("#### Split Counts")
        for split_name, count, color, pct_label in [
            ("Train Set",        193474, "#10b981", "73.7%"),
            ("Validation Set",    27632, "#38bdf8", "10.5%"),
            ("Test Set",          13807, "#a78bfa", " 5.3%"),
            ("External Val Set",  27630, "#fb923c", "10.5%"),
        ]:
            pct = count / 262543 * 100
            st.markdown(
                f"<div style='"
                f"padding:11px 16px;"
                f"border-radius:9px;"
                f"background:linear-gradient(90deg,rgba(15,23,42,0.8),rgba(30,41,59,0.5));"
                f"border-left:4px solid {color};"
                f"border:1px solid {color}55;"
                f"border-left:4px solid {color};"
                f"margin-bottom:8px;"
                f"display:flex;justify-content:space-between;align-items:center;"
                f"'>"
                f"<span style='color:#cbd5e1;font-size:0.84rem;'>{split_name}</span>"
                f"<span style='font-weight:800;color:{color};font-family:Outfit,sans-serif;font-size:1.0rem;'>"
                f"{count:,}&nbsp;<span style='color:#64748b;font-size:0.7rem;font-weight:500;'>({pct:.1f}%)</span></span>"
                f"</div>",
                unsafe_allow_html=True,
            )
    with ds_col2:
        st.markdown("#### Domain Coverage")
        st.dataframe(pd.DataFrame([
            {"Domain": "bundestag_speech", "Description": "Speeches in German Bundestag",          "Format": "Length-matched"},
            {"Domain": "europarl_speech",  "Description": "Speeches in European Parliament",        "Format": "Length-matched"},
            {"Domain": "state_law",        "Description": "German legislative / admin. law texts",  "Format": "Length-matched"},
        ]), use_container_width=True, hide_index=True)

    # ── Live full-suite classification ─────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🧪 Live Classification — All Models")
    st.markdown("Paste any German text to instantly classify it with every trained model and see a majority verdict.")

    live_text = st.text_area(
        "Text for full-suite analysis:",
        height=140,
        placeholder="Geben Sie hier Ihren deutschen Text ein...",
        key="export_live_text",
    )
    run_full_analysis = st.button(
        "Run Full Suite Analysis 🚀", type="primary",
        use_container_width=True, key="export_analyze_btn",
    )

    if run_full_analysis and live_text.strip() and AITextDetectorClass is not None:
        st.info(
            "⏳ Running 7 models **sequentially** (one GPU slot at a time). "
            "Each model is offloaded from VRAM after inference to prevent CUDA OOM. "
            "Expected time: ~30–60 seconds."
        )
        with st.spinner("Running sequential GPU-safe inference across all models..."):
            analysis_rows = predict_all_models_sequentially(live_text, threshold)

        valid    = [r for r in analysis_rows if "Error" not in r["Verdict"]]
        ai_votes = sum(1 for r in valid if "AI" in r["Verdict"])
        hm_votes = len(valid) - ai_votes
        majority = "🤖 AI-Generated" if ai_votes > hm_votes else "👤 Human-Written"
        maj_col  = "#f43f5e" if "AI" in majority else "#10b981"
        st.markdown(
            f"<div style='padding:14px 20px;margin:10px 0 14px;border-radius:12px;"
            f"background:rgba(30,41,59,0.6);border:2px solid {maj_col}44;'>"
            f"<span style='font-size:0.75rem;color:#94a3b8;font-weight:600;text-transform:uppercase;'>Majority Verdict</span>"
            f"<div style='font-size:1.55rem;font-weight:800;color:{maj_col};font-family:Outfit,sans-serif;'>{majority}</div>"
            f"<div style='font-size:0.83rem;color:#cbd5e1;margin-top:3px;'>"
            f"{ai_votes} / {len(valid)} models → <b>AI</b> &nbsp;|&nbsp; {hm_votes} / {len(valid)} models → <b>Human</b></div>"
            f"</div>",
            unsafe_allow_html=True,
        )
        display_rows = [{k: v for k, v in r.items() if k != "key"} for r in analysis_rows]
        st.dataframe(pd.DataFrame(display_rows), use_container_width=True, hide_index=True)
    elif run_full_analysis:
        st.warning("Please enter some text above before running the analysis.")

    # ── Download section ────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### ⬇️ Download Combined Multi-Model Report")

    export_payload = {
        "project":              "German AI Text Detection GBERT-large Suite",
        "date_exported":        "2026-07-09",
        "calibrated_threshold": threshold,
        "system_configuration": {
            "GPU Hardware": "NVIDIA GeForce RTX 4080 (16 GB VRAM)",
            "Precision":    "BF16 (CUDA BFloat16)",
            "Base Model":   "deepset/gbert-large",
            "Dimensions":   1024,
        },
        "leaderboard":      LEADERBOARD_DATA,
        "model_details":    {k: {**v, **MODEL_DETAILS.get(k, {})} for k, v in AVAILABLE_MODELS.items()},
    }

    try:
        json_str = json.dumps(export_payload, indent=2, ensure_ascii=False)
        with open("gbert_training_export.json", "w", encoding="utf-8") as f:
            f.write(json_str)

        rows = []
        for m_key, m_info in AVAILABLE_MODELS.items():
            d = MODEL_DETAILS.get(m_key, {})
            rows.append({
                "model_key":    m_key,
                "model_name":   m_info["name"],
                "model_path":   m_info["path"],
                "accuracy":     m_info["accuracy"],
                "macro_f1":     m_info["f1"],
                "ood_f1":       d.get("ood_f1", "N/A"),
                "val_f1":       d.get("val_f1", "N/A"),
                "train_loss":   d.get("train_loss", "N/A"),
                "dataset":      d.get("dataset", "N/A"),
                "epochs":       d.get("epochs", 3),
                "batch_size":   d.get("batch", 16),
                "learning_rate":d.get("lr", "2e-5"),
                "notes":        d.get("notes", ""),
            })

        csv_buf = io.StringIO()
        writer  = csv.DictWriter(csv_buf, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
        csv_str = csv_buf.getvalue()
        with open("gbert_training_logs.csv", "w", encoding="utf-8", newline="") as f:
            f.write(csv_str)

        st.markdown(
            """
            <div style="background:rgba(16,185,129,0.08);border:1px solid rgba(16,185,129,0.25);
                        border-radius:12px;padding:16px;margin-bottom:18px;">
                <h5 style="color:#10b981;margin:0 0 8px;font-family:'Outfit',sans-serif;font-weight:600;">
                    📂 Local Copies Updated!
                </h5>
                <ul style="margin:4px 0 0 18px;padding:0;font-size:0.87rem;color:#a5b4fc;font-family:monospace;">
                    <li>gbert_training_export.json — full suite JSON report</li>
                    <li>gbert_training_logs.csv — per-model metrics CSV</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
    except Exception as e:
        st.warning(f"Could not save local copies: {e}")

    dl_col1, dl_col2 = st.columns(2)
    with dl_col1:
        st.download_button(
            label="⬇️ Download Suite JSON Report",
            data=json_str,
            file_name="gbert_multi_model_report.json",
            mime="application/json",
            use_container_width=True,
            type="primary",
        )
    with dl_col2:
        st.download_button(
            label="⬇️ Download Suite Summary CSV",
            data=csv_str.encode("utf-8"),
            file_name="gbert_training_logs.csv",
            mime="text/csv",
            use_container_width=True,
        )
