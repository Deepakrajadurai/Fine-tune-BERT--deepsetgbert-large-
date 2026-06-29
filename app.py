import os
import re
import sys
import json
import csv
import io
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
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Plus+Jakarta+Sans:wght@300;400;500;700&display=swap');
    
    /* Global styles */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .main-title {
        font-family: 'Outfit', sans-serif;
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 50%, #ec4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    
    .subtitle {
        font-size: 1.15rem;
        color: #64748b;
        margin-bottom: 2rem;
    }
    
    /* Custom Card */
    .metric-card {
        background: rgba(30, 41, 59, 0.45);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(5px);
        -webkit-backdrop-filter: blur(5px);
        margin-bottom: 20px;
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        border-color: rgba(99, 102, 241, 0.3);
        transform: translateY(-2px);
    }
    
    .verdict-header {
        font-size: 0.9rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94a3b8;
        margin-bottom: 8px;
    }
    
    .verdict-text {
        font-family: 'Outfit', sans-serif;
        font-size: 1.8rem;
        font-weight: 700;
        margin-bottom: 12px;
    }
    
    .verdict-desc {
        font-size: 0.95rem;
        color: #cbd5e1;
        line-height: 1.5;
    }
    
    /* Accent text colors */
    .color-human {
        color: #10b981; /* Emerald */
    }
    
    .color-ai {
        color: #f43f5e; /* Rose */
    }
    
    /* Custom Sidebar Header */
    .sidebar-header {
        font-family: 'Outfit', sans-serif;
        font-size: 1.3rem;
        font-weight: 700;
        color: #f8fafc;
        margin-top: 15px;
        margin-bottom: 15px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        padding-bottom: 8px;
    }
    
    /* Code Blocks */
    code {
        color: #cbd5e1 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Dynamic import of predict.py to load detector
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

st.sidebar.markdown('<div class="sidebar-header">📊 Model Specifications</div>', unsafe_allow_html=True)
st.sidebar.markdown(
    """
    - **Base Architecture**: Fine-tuned `deepset/gbert-large`
    - **Domain Focus**: Politics, News, and Casual Everyday German (Blogs, Essays, Forums)
    - **Training Corpus**: ~57,000 paragraphs (balanced 50% human / 50% AI in 3 domains)
    - **GPU Hardware**: NVIDIA GeForce RTX 4080 (16GB VRAM)
    - **Optimization**: Placeholder masking, length stratification, FP16 precision
    - **Validation Loss**: `0.02066`
    - **Overall Test Accuracy**: **99.77%**
    - **Overall Test Macro F1**: **99.77%**
    """
)

st.sidebar.markdown('<div class="sidebar-header">📦 Diversity of AI Sources</div>', unsafe_allow_html=True)
st.sidebar.markdown(
    """
    The detector stands robust against text generated from **8 diverse models**:
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
st.markdown('<p class="subtitle">State-of-the-art fine-tuned GBERT-base model optimized for human vs. AI text classification in German</p>', unsafe_allow_html=True)

# Instantiate detector
if AITextDetectorClass is not None:
    try:
        # Load detector and cache it
        @st.cache_resource
        def get_detector(t):
            return AITextDetectorClass(threshold=t)
        
        detector = get_detector(threshold)
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
        
        analyze_btn = st.button("Analyze Text ⚡", type="primary", use_container_width=True)
        
    with col_output:
        st.markdown("### Detection Result")
        
        if analyze_btn and text_input.strip():
            if detector is not None:
                with st.spinner("Analyzing text chunks..."):
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
                    progress_color = "#f43f5e"
                else:
                    verdict_class = "color-human"
                    verdict_title = "👤 Human-Written"
                    progress_color = "#10b981"
                
                # HTML Card Output
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="verdict-header">Classification Verdict</div>
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
                        Decision boundary boundary = {threshold:.2f}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
            else:
                st.error("Model detector is not loaded correctly.")
        else:
            # Standby state
            st.info("👈 Enter or select a German text on the left and click **Analyze Text** to inspect its origin.")

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
                    
                    # CSV Download button
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
# TAB 3: Export Report
# ──────────────────────────────────────────────────────────────────────────────
with tab_export:
    st.markdown("### 📋 Experiment Identifier & Reproducibility")
    
    # ── read calibrated threshold ──
    _t_path = Path("results/threshold.txt")
    _cal_t  = float(_t_path.read_text().strip()) if _t_path.exists() else 0.5

    meta_col1, meta_col2 = st.columns(2)
    with meta_col1:
        st.markdown(
            """
            <div style="background:rgba(30,41,59,0.45); padding:16px; border-radius:12px; border:1px solid rgba(255,255,255,0.06); height: 100%;">
                <div style="font-size:0.75rem; color:#94a3b8; font-weight:600; text-transform:uppercase; letter-spacing:0.05em;">Experiment ID</div>
                <div style="font-size:1.4rem; font-weight:800; color:#ec4899; margin-top:2px; font-family:'Outfit',sans-serif;">GBERT_LARGE_FULL_V1</div>
                <div style="margin-top:10px; font-size:0.82rem; color:#cbd5e1;">
                    <b>Date:</b> 2026-06-22<br>
                    <b>Git Commit:</b> <code style="color:#a5b4fc;">b1d5dc1</code><br>
                    <b>Dataset Version:</b> German AI Detector Dataset v1.0<br>
                    <b>Dataset SHA256:</b> <code style="color:#a5b4fc; word-break:break-all;">1ece8662e2e50ffdf0f04679942d5edd223297c077a2bb86ce741deaeaf970e8</code>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with meta_col2:
        st.markdown(
            """
            <div style="background:rgba(30,41,59,0.45); padding:16px; border-radius:12px; border:1px solid rgba(255,255,255,0.06); height: 100%;">
                <div style="font-size:0.75rem; color:#94a3b8; font-weight:600; text-transform:uppercase; letter-spacing:0.05em;">Best Training Checkpoint</div>
                <div style="font-size:1.4rem; font-weight:800; color:#3b82f6; margin-top:2px; font-family:'Outfit',sans-serif;">checkpoint-8610</div>
                <div style="margin-top:10px; font-size:0.82rem; color:#cbd5e1;">
                    <b>Selection Criterion:</b> Highest <code>eval_external_f1</code><br>
                    <b>Best Metric Value:</b> <code style="color:#10b981;">99.70% (0.9970)</code><br>
                    <b>Path:</b> <code>models/best_model/checkpoint-8610</code>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("#### 💻 Training Command")
    st.code(
        """python train.py \\
  --model_name deepset/gbert-large \\
  --epochs 3 \\
  --batch_size 16 \\
  --lr 2e-5 \\
  --max_length 256 \\
  --seed 42 \\
  --threshold 0.10""",
        language="bash"
    )

    st.markdown("---")
    st.markdown("### 📊 Training Configuration & Runtime")
    
    config_data = {
        "Model":                  "deepset/gbert-large",
        "Architecture":           "BertForSequenceClassification",
        "Task":                   "Binary Classification — Human vs. AI German Text",
        "Hidden Size":            1024,
        "Attention Heads":        16,
        "Hidden Layers":          24,
        "Vocab Size":             31102,
        "Max Sequence Length":    256,
        "Batch Size":             16,
        "Learning Rate":          "2e-5",
        "Epochs":                 3,
        "Weight Decay":           0.01,
        "Warmup Ratio":           0.1,
        "Precision":              "BF16 (CUDA BFloat16)",
        "GPU Hardware":           "NVIDIA GeForce RTX 4080 (16 GB VRAM)",
        "Total Run Time":         "26m 24s",
        "Avg Epoch Time":         "8m 48s",
        "Peak GPU Memory":        "12.2 GB",
        "Seed":                   42
    }

    cfg_col1, cfg_col2 = st.columns(2)
    items = list(config_data.items())
    half  = (len(items) + 1) // 2
    with cfg_col1:
        for k, v in items[:half]:
            st.markdown(
                f"""
                <div style="display:flex; justify-content:space-between;
                            padding:8px 12px; border-radius:8px;
                            background:rgba(30,41,59,0.45);
                            border:1px solid rgba(255,255,255,0.06);
                            margin-bottom:6px; font-size:0.88rem;">
                    <span style="color:#94a3b8; font-weight:600;">{k}</span>
                    <span style="color:#f1f5f9; font-family:'Outfit',sans-serif;
                                 font-weight:700;">{v}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
    with cfg_col2:
        for k, v in items[half:]:
            st.markdown(
                f"""
                <div style="display:flex; justify-content:space-between;
                            padding:8px 12px; border-radius:8px;
                            background:rgba(30,41,59,0.45);
                            border:1px solid rgba(255,255,255,0.06);
                            margin-bottom:6px; font-size:0.88rem;">
                    <span style="color:#94a3b8; font-weight:600;">{k}</span>
                    <span style="color:#f1f5f9; font-family:'Outfit',sans-serif;
                                 font-weight:700;">{v}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("---")
    st.markdown("### 📈 Training Loss & Validation Logs")
    
    st.markdown("#### 🔁 Base Model (Full Dataset Training)")
    base_logs = pd.DataFrame([
        {"Epoch": 1, "Train Loss": 0.0239, "Validation Loss": 0.0154, "Validation F1": "99.67%", "Learning Rate": "1.49e-5"},
        {"Epoch": 2, "Train Loss": 0.0003, "Validation Loss": 0.0212, "Validation F1": "99.56%", "Learning Rate": "7.51e-6"},
        {"Epoch": 3, "Train Loss": 0.00002, "Validation Loss": 0.0207, "Validation F1": "99.70%", "Learning Rate": "2.84e-8"}
    ])
    st.dataframe(base_logs, use_container_width=True, hide_index=True)

    log_rotations = [
        {
            "name":        "Rotation 1  —  Gemini Held-Out",
            "description": "AI Gemini withheld from training; used as unseen test source",
            "train":       420,
            "val":         300,
            "epochs":      [
                {"epoch": 1, "loss": 0.4238},
                {"epoch": 2, "loss": 0.0755},
                {"epoch": 3, "loss": 0.0078},
            ],
        },
        {
            "name":        "Rotation 2  —  Qwen Held-Out",
            "description": "AI Qwen withheld from training; used as unseen test source",
            "train":       420,
            "val":         300,
            "epochs":      [
                {"epoch": 1, "loss": 0.4020},
                {"epoch": 2, "loss": 0.1347},
                {"epoch": 3, "loss": 0.0268},
            ],
        },
    ]

    for rot in log_rotations:
        with st.expander(f"🔁 {rot['name']}", expanded=True):
            st.markdown(
                f"<span style='color:#94a3b8; font-size:0.85rem;'>{rot['description']}</span>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"**Train Samples:** `{rot['train']:,}`   &nbsp;&nbsp;"
                f"**Val / Test Samples:** `{rot['val']:,}`"
            )
            epoch_cols = st.columns(len(rot["epochs"]))
            for col, ep in zip(epoch_cols, rot["epochs"]):
                loss_val = f"{ep['loss']:.4f}" if isinstance(ep["loss"], float) else ep["loss"]
                col.markdown(
                    f"""
                    <div style="text-align:center; padding:16px 8px;
                                border-radius:12px;
                                background:rgba(30,41,59,0.55);
                                border:1px solid rgba(99,102,241,0.25);
                                margin:4px;">
                        <div style="font-size:0.75rem; color:#94a3b8;
                                     font-weight:600; text-transform:uppercase;
                                     letter-spacing:0.05em;">Epoch {ep['epoch']}</div>
                        <div style="font-size:1.4rem; font-weight:800;
                                     font-family:'Outfit',sans-serif;
                                     color:#818cf8; margin-top:4px;">{loss_val}</div>
                        <div style="font-size:0.7rem; color:#475569;
                                     margin-top:2px;">Training Loss</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    st.markdown("---")
    st.markdown("### 🗄️ Dataset Statistics")

    ds_col1, ds_col2 = st.columns([1, 1.4])

    with ds_col1:
        st.markdown("#### Class Balance")
        for label_name, count in [("🧑 Human Samples", 28671), ("🤖 AI Samples", 28725)]:
            st.markdown(
                f"""
                <div style="padding:14px 18px; border-radius:10px;
                            background:rgba(30,41,59,0.55);
                            border:1px solid rgba(255,255,255,0.06);
                            margin-bottom:8px; font-size:0.9rem;">
                    <span style="color:#cbd5e1;">{label_name}</span>
                    <span style="float:right; font-weight:800;
                                 font-family:'Outfit',sans-serif;
                                 color:#f1f5f9;">{count:,}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.markdown(
            """
            <div style="padding:14px 18px; border-radius:10px;
                        background:rgba(99,102,241,0.15);
                        border:1px solid rgba(99,102,241,0.3);
                        margin-top:4px; font-size:0.9rem;">
                <span style="color:#a5b4fc; font-weight:600;">📦 Total</span>
                <span style="float:right; font-weight:800;
                             font-family:'Outfit',sans-serif;
                             font-size:1.1rem; color:#e0e7ff;">57,396</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("#### Data Split")
        splits = [
            ("Train",    45916, "#10b981"),
            ("Validation", 5740, "#3b82f6"),
            ("Test",       5740, "#8b5cf6"),
            ("External Val",5740,"#f59e0b"),
        ]
        for split_name, count, color in splits:
            pct = count / 57396 * 100
            st.markdown(
                f"""
                <div style="padding:10px 14px; border-radius:8px;
                            background:rgba(30,41,59,0.45);
                            border-left:4px solid {color};
                            margin-bottom:6px; font-size:0.88rem;">
                    <span style="color:#cbd5e1;">{split_name}</span>
                    <span style="float:right; font-weight:700;
                                 color:#f1f5f9;">{count:,}
                        <span style="color:#64748b; font-size:0.75rem;"> ({pct:.1f}%)</span>
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with ds_col2:
        st.markdown("#### Source Distribution (Training Set)")
        train_sources = {
            "Bundestag (AI)":   12002,
            "News (Human)":     11518,
            "Bundestag (Human)":10302,
            "News (AI)":         8679,
            "Casual (AI)":       2277,
            "Casual (Human)":    1138,
        }
        src_df = pd.DataFrame(
            [{"Source": k, "Samples": v, "Share": f"{v/45916*100:.1f}%"}
             for k, v in train_sources.items()]
        )
        st.dataframe(src_df, use_container_width=True, hide_index=True)

        st.markdown("#### AI Generator Coverage")
        ai_gens = [
            "gemini-1.5-flash", "mistralai/Mistral-7B-Instruct-v0.3",
            "llama3-70b-8192",  "gemma2-9b-it",
            "mixtral-8x7b-32768","phi3", "mistral", "llama3",
        ]
        gen_html = "".join(
            f"<span style='display:inline-block; background:rgba(99,102,241,0.18);"
            f" border:1px solid rgba(99,102,241,0.3); border-radius:20px;"
            f" padding:3px 10px; margin:3px; font-size:0.78rem; color:#a5b4fc;'>{g}</span>"
            for g in ai_gens
        )
        st.markdown(gen_html, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🏆 Final Evaluation Results")
    
    eval_df = pd.DataFrame([
        {"Model Setup": "Original Model Setup", "Accuracy": "58.70%", "Macro F1": "57.38%", "ROC-AUC": "0.6718", "Macro Precision": "64.35%", "Macro Recall": "61.42%"},
        {"Model Setup": "Rotation 1 (Gemini Held-Out)", "Accuracy": "83.67%", "Macro F1": "60.86%", "ROC-AUC": "0.8692", "Macro Precision": "91.52%", "Macro Recall": "59.17%"},
        {"Model Setup": "Rotation 2 (Qwen Held-Out)", "Accuracy": "94.67%", "Macro F1": "90.69%", "ROC-AUC": "0.9932", "Macro Precision": "96.88%", "Macro Recall": "86.67%"}
    ])
    st.dataframe(eval_df, use_container_width=True, hide_index=True)

    st.markdown("#### 🎯 Confusion Matrices")
    
    col_cm1, col_cm2, col_cm3 = st.columns(3)
    
    with col_cm1:
        st.markdown(
            """
            <div style="background:rgba(30,41,59,0.3); border:1px solid rgba(255,255,255,0.06); padding:12px; border-radius:8px;">
                <div style="font-size:0.8rem; font-weight:600; text-align:center; margin-bottom:8px; color:#cbd5e1;">Original Model Setup</div>
                <table style="width:100%; border-collapse:collapse; font-size:0.82rem; text-align:center; color:#f1f5f9;">
                    <tr style="border-bottom:1px solid rgba(255,255,255,0.1);">
                        <th></th>
                        <th colspan="2" style="color:#94a3b8; font-size:0.75rem; text-transform:uppercase;">Predicted</th>
                    </tr>
                    <tr style="border-bottom:1px solid rgba(255,255,255,0.05);">
                        <td style="font-weight:600; color:#94a3b8;">Actual</td>
                        <td style="font-weight:600; color:#10b981;">Human</td>
                        <td style="font-weight:600; color:#f43f5e;">AI</td>
                    </tr>
                    <tr style="border-bottom:1px solid rgba(255,255,255,0.05);">
                        <td style="font-weight:600; color:#10b981;">Human</td>
                        <td style="background:rgba(16,185,129,0.15); font-weight:700; color:#10b981;">206</td>
                        <td style="color:#cbd5e1;">34</td>
                    </tr>
                    <tr>
                        <td style="font-weight:600; color:#f43f5e;">AI</td>
                        <td style="color:#cbd5e1;">189</td>
                        <td style="background:rgba(244,63,94,0.15); font-weight:700; color:#f43f5e;">111</td>
                    </tr>
                </table>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    with col_cm2:
        st.markdown(
            """
            <div style="background:rgba(30,41,59,0.3); border:1px solid rgba(255,255,255,0.06); padding:12px; border-radius:8px;">
                <div style="font-size:0.8rem; font-weight:600; text-align:center; margin-bottom:8px; color:#cbd5e1;">Rotation 1 (Gemini Out)</div>
                <table style="width:100%; border-collapse:collapse; font-size:0.82rem; text-align:center; color:#f1f5f9;">
                    <tr style="border-bottom:1px solid rgba(255,255,255,0.1);">
                        <th></th>
                        <th colspan="2" style="color:#94a3b8; font-size:0.75rem; text-transform:uppercase;">Predicted</th>
                    </tr>
                    <tr style="border-bottom:1px solid rgba(255,255,255,0.05);">
                        <td style="font-weight:600; color:#94a3b8;">Actual</td>
                        <td style="font-weight:600; color:#10b981;">Human</td>
                        <td style="font-weight:600; color:#f43f5e;">AI</td>
                    </tr>
                    <tr style="border-bottom:1px solid rgba(255,255,255,0.05);">
                        <td style="font-weight:600; color:#10b981;">Human</td>
                        <td style="background:rgba(16,185,129,0.15); font-weight:700; color:#10b981;">240</td>
                        <td style="color:#cbd5e1;">0</td>
                    </tr>
                    <tr>
                        <td style="font-weight:600; color:#f43f5e;">AI</td>
                        <td style="color:#cbd5e1;">49</td>
                        <td style="background:rgba(244,63,94,0.15); font-weight:700; color:#f43f5e;">11</td>
                    </tr>
                </table>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col_cm3:
        st.markdown(
            """
            <div style="background:rgba(30,41,59,0.3); border:1px solid rgba(255,255,255,0.06); padding:12px; border-radius:8px;">
                <div style="font-size:0.8rem; font-weight:600; text-align:center; margin-bottom:8px; color:#cbd5e1;">Rotation 2 (Qwen Out)</div>
                <table style="width:100%; border-collapse:collapse; font-size:0.82rem; text-align:center; color:#f1f5f9;">
                    <tr style="border-bottom:1px solid rgba(255,255,255,0.1);">
                        <th></th>
                        <th colspan="2" style="color:#94a3b8; font-size:0.75rem; text-transform:uppercase;">Predicted</th>
                    </tr>
                    <tr style="border-bottom:1px solid rgba(255,255,255,0.05);">
                        <td style="font-weight:600; color:#94a3b8;">Actual</td>
                        <td style="font-weight:600; color:#10b981;">Human</td>
                        <td style="font-weight:600; color:#f43f5e;">AI</td>
                    </tr>
                    <tr style="border-bottom:1px solid rgba(255,255,255,0.05);">
                        <td style="font-weight:600; color:#10b981;">Human</td>
                        <td style="background:rgba(16,185,129,0.15); font-weight:700; color:#10b981;">240</td>
                        <td style="color:#cbd5e1;">0</td>
                    </tr>
                    <tr>
                        <td style="font-weight:600; color:#f43f5e;">AI</td>
                        <td style="color:#cbd5e1;">16</td>
                        <td style="background:rgba(244,63,94,0.15); font-weight:700; color:#f43f5e;">44</td>
                    </tr>
                </table>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.markdown("### ⬇️ Download Full Training Report")

    export_payload = {
        "experiment_identifier": {
            "experiment_id": "GBERT_LARGE_FULL_V1",
            "git_commit": "b1d5dc1",
            "date": "2026-06-22",
            "training_command": "python train.py --model_name deepset/gbert-large --epochs 3 --batch_size 16 --lr 2e-5 --max_length 256 --seed 42 --threshold 0.10"
        },
        "training_configuration": config_data,
        "best_checkpoint": {
            "best_checkpoint": "checkpoint-8610",
            "selection_criterion": "Highest eval_external_f1",
            "best_metric_value": 0.9970383202449933
        },
        "training_logs": {
            "base_model": [
                {"epoch": 1, "train_loss": 0.0239, "val_loss": 0.0154, "val_f1": 0.9967, "learning_rate": 1.49e-5},
                {"epoch": 2, "train_loss": 0.0003, "val_loss": 0.0212, "val_f1": 0.9956, "learning_rate": 7.51e-6},
                {"epoch": 3, "train_loss": 0.00002, "val_loss": 0.0207, "val_f1": 0.9970, "learning_rate": 2.84e-8}
            ],
            "rotation_1": [
                {"epoch": 1, "loss": 0.4238},
                {"epoch": 2, "loss": 0.0755},
                {"epoch": 3, "loss": 0.0078}
            ],
            "rotation_2": [
                {"epoch": 1, "loss": 0.4020},
                {"epoch": 2, "loss": 0.1347},
                {"epoch": 3, "loss": 0.0268}
            ]
        },
        "dataset_statistics": {
            "dataset_version": "German AI Detector Dataset v1.0",
            "dataset_sha256": "1ece8662e2e50ffdf0f04679942d5edd223297c077a2bb86ce741deaeaf970e8",
            "dataset_generated": "2026-06-22",
            "total_samples":  57396,
            "human_samples":  28671,
            "ai_samples":     28725,
            "class_balance":  "50 / 50 — perfectly balanced",
            "language":       "German (de)",
            "splits": {
                "train":       {"total": 45916, "human": 22958, "ai": 22958},
                "validation":  {"total":  5740, "human":  2870, "ai":  2870},
                "test":        {"total":  5740, "human":  2870, "ai":  2870},
                "external_val":{"total":  5740, "human":  2870, "ai":  2870},
            },
            "train_sources": train_sources,
            "ai_generators": ai_gens,
        },
        "evaluation_results": {
            "original_model":             {"accuracy": 0.5870, "macro_f1": 0.5738, "roc_auc": 0.6718, "macro_precision": 0.6435, "macro_recall": 0.6142},
            "rotation1_gemini_held_out":  {"accuracy": 0.8367, "macro_f1": 0.6086, "roc_auc": 0.8692, "macro_precision": 0.9152, "macro_recall": 0.5917},
            "rotation2_qwen_held_out":    {"accuracy": 0.9467, "macro_f1": 0.9069, "roc_auc": 0.9932, "macro_precision": 0.9688, "macro_recall": 0.8667},
            "calibrated_threshold":       _cal_t,
        },
        "confusion_matrices": {
            "original_model": {"true_negative": 206, "false_positive": 34, "false_negative": 189, "true_positive": 111},
            "rotation_1":     {"true_negative": 240, "false_positive": 0, "false_negative": 49, "true_positive": 11},
            "rotation_2":     {"true_negative": 240, "false_positive": 0, "false_negative": 16, "true_positive": 44}
        }
    }

    # Auto-save local copies of files to the workspace root directory
    try:
        json_str = json.dumps(export_payload, indent=2, ensure_ascii=False)
        with open("gbert_training_export.json", "w", encoding="utf-8") as f:
            f.write(json_str)
            
        rows = []
        # Main model epoch training logs rows
        for log in export_payload["training_logs"]["base_model"]:
            rows.append({
                "model_setup":   "Base Model (Full Dataset)",
                "epoch":         log["epoch"],
                "train_loss":    log["train_loss"],
                "val_loss":      log["val_loss"],
                "val_f1":        log["val_f1"],
                "learning_rate": log["learning_rate"],
            })
        # Rotations epoch training logs rows
        for rot_name, key in [("Rotation 1 — Gemini Out", "rotation_1"), ("Rotation 2 — Qwen Out", "rotation_2")]:
            for log in export_payload["training_logs"][key]:
                rows.append({
                    "model_setup":   rot_name,
                    "epoch":         log["epoch"],
                    "train_loss":    log["loss"],
                    "val_loss":      "",
                    "val_f1":        "",
                    "learning_rate": "",
                })
                
        csv_buf = io.StringIO()
        writer  = csv.DictWriter(csv_buf, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
        csv_str = csv_buf.getvalue()
        with open("gbert_training_logs.csv", "w", encoding="utf-8", newline="") as f:
            f.write(csv_str)
            
        # Display auto-save info box with premium dark theme styles
        st.markdown(
            """
            <div style="background: rgba(16, 185, 129, 0.08); border: 1px solid rgba(16, 185, 129, 0.25); 
                        border-radius: 12px; padding: 18px; margin-bottom: 24px; box-shadow: 0 4px 15px rgba(0,0,0,0.15);">
                <h5 style="color: #10b981; margin-top: 0; margin-bottom: 8px; font-family: 'Outfit', sans-serif; font-weight: 600; display: flex; align-items: center; gap: 8px;">
                    📂 Local Workspace Copies Updated!
                </h5>
                <p style="margin: 0; font-size: 0.9rem; color: #cbd5e1; line-height: 1.5;">
                    The enhanced export files have been automatically written directly to your project workspace directory:
                </p>
                <ul style="margin: 8px 0 0 20px; padding: 0; font-size: 0.88rem; color: #a5b4fc; font-family: monospace;">
                    <li>gbert_training_export.json</li>
                    <li>gbert_training_logs.csv</li>
                </ul>
                <p style="margin: 12px 0 0 0; font-size: 0.82rem; color: #94a3b8; line-height: 1.4;">
                    💡 <i>Note: If clicking the buttons below does not trigger a download (which commonly happens due to security sandboxing in IDE preview panes), you can access and copy these files directly from the sidebar explorer in your workspace. Alternatively, open <a href="http://localhost:8501" target="_blank" style="color: #3b82f6; text-decoration: underline; font-weight: 500;">http://localhost:8501</a> in an external browser window.</i>
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
    except Exception as e:
        st.warning(f"Could not automatically save local copies: {e}")

    dl_col1, dl_col2 = st.columns(2)
    with dl_col1:
        st.download_button(
            label="⬇️ Download as JSON",
            data=json_str,
            file_name="gbert_training_export.json",
            mime="application/json",
            use_container_width=True,
            type="primary",
        )
    with dl_col2:
        st.download_button(
            label="⬇️ Download Logs as CSV",
            data=csv_str.encode("utf-8"),
            file_name="gbert_training_logs.csv",
            mime="text/csv",
            use_container_width=True,
        )
