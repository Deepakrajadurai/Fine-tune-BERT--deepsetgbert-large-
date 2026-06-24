import os
import re
import sys
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
        calibrated_t = float(f.read_text().strip() if hasattr(f, 'read_text') else f.read().strip())
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
tab_single, tab_batch = st.tabs(["📝 Single Text Analysis", "📁 Batch File Upload"])

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
