"""
NLP Graphic Tool - Research Assignment Option 2

A modern Streamlit application implementing all professor requirements:
- 2.1 Eliminate stopwords
- 2.2 Lemmatize terms
- 2.3 Compute frequencies
- 2.4 Measure distances from strategic points (start and end)
- 2.5 Compute compound relevance indices (50% frequency + 50% earliness)

Supports Tigrinya and English text.
"""

import streamlit as st
import pandas as pd
import json
from pathlib import Path

import plotly.express as px
import plotly.graph_objects as go
from nlp_pipeline import (
    run_pipeline, TIGRINYA_AVAILABLE, NLTK_AVAILABLE, LANGDETECT_AVAILABLE,
    PipelineResult, _detect_language
)

# Path to sample fixtures
FIXTURES_DIR = Path(__file__).parent / "tests" / "fixtures"


def load_sample(name: str) -> str:
    """Load sample text from fixtures."""
    path = FIXTURES_DIR / name
    if path.exists():
        return path.read_text(encoding="utf-8")
    return ""

# Page config
st.set_page_config(
    page_title="NLP Graphic Tool | Research Assignment",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern look
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    .main {
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%);
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background: transparent;
    }
    
    h1, h2, h3 {
        color: #f8fafc !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em;
    }
    
    /* Enhance buttons to look more premium */
    button[kind="primary"] {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%) !important;
        border: none !important;
        box-shadow: 0 4px 14px 0 rgba(124, 58, 237, 0.39) !important;
        transition: all 0.2s ease !important;
    }
    button[kind="primary"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(124, 58, 237, 0.5) !important;
    }

    /* Glassmorphism containers */
    .metric-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 24px;
        margin: 12px 0;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
    }
    
    .requirement-badge {
        display: inline-block;
        background: linear-gradient(135deg, #38bdf8 0%, #3b82f6 100%);
        color: white;
        padding: 4px 14px;
        border-radius: 24px;
        font-size: 0.85em;
        font-weight: 600;
        margin: 3px;
        box-shadow: 0 2px 10px rgba(59, 130, 246, 0.3);
    }
    
    /* Text area styling */
    .stTextArea textarea {
        background-color: rgba(15, 23, 42, 0.6) !important;
        border: 1px solid rgba(148, 163, 184, 0.2) !important;
        border-radius: 12px !important;
        color: #f1f5f9 !important;
        font-size: 1.05rem !important;
        padding: 1rem !important;
    }
    .stTextArea textarea:focus {
        border-color: #8b5cf6 !important;
        box-shadow: 0 0 0 1px #8b5cf6 !important;
    }
    
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        font-weight: 700 !important;
        color: #a5b4fc !important;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.title("📊 NLP Graphic Tool")
st.markdown("""
<div class="requirement-badge">2.1 Stopwords</div>
<div class="requirement-badge">2.2 Lemmatization</div>
<div class="requirement-badge">2.3 Frequencies</div>
<div class="requirement-badge">2.4 Distances</div>
<div class="requirement-badge">2.5 Compound Relevance</div>
""", unsafe_allow_html=True)
st.markdown("*Research-Oriented Assignment Option 2 — Tigrinya & English Support*")
st.divider()

# Input mode (needed before load buttons)
# Test Samples Setup
sample_options = {
    "None (Enter own text)": "",
    "Tigrinya Sample 1": "sample_tigrinya.txt",
    "Tigrinya Sample 2 (Long)": "sample_tigrinya_long.txt",
    "English Sample": "sample_english.txt"
}

def load_sample_to_textarea():
    """Callback to load sample text directly into the text area's session state."""
    sample_file = sample_options[st.session_state["sample_dropdown"]]
    if sample_file:
        text = load_sample(sample_file)
        if text:
            st.session_state["text_area_widget"] = text
        else:
            st.warning(f"Sample file {sample_file} not found in tests/fixtures/")
    else:
        st.session_state["text_area_widget"] = ""

with st.sidebar:
    st.header("⚙️ Settings & Test Data")
    
    # Dropdown for test samples using on_change callback
    selected_sample_name = st.selectbox(
        "Load Test Sample", 
        list(sample_options.keys()),
        key="sample_dropdown",
        on_change=load_sample_to_textarea
    )
    
    st.divider()
    st.subheader("Pipeline Options")
    auto_detect_lang = st.checkbox("Auto-detect Language", True, help="Automatically detect between English and Tigrinya based on text content")
    
    # Only show manual language selection if auto-detect is off
    language = "auto"
    if not auto_detect_lang:
        language = st.selectbox(
            "Manual Language Override",
            ["tigrinya", "english"],
            format_func=lambda x: "Tigrinya (ትግርኛ)" if x == "tigrinya" else "English"
        )
    
    remove_stopwords = st.checkbox("Eliminate stopwords (2.1)", True)
    lemmatize = st.checkbox("Lemmatize terms (2.2)", True)
    st.divider()
    st.caption("Language is auto-detected using Ethiopic script analysis + langdetect library.")

# Input
st.subheader("📝 Input Text")

input_mode = st.radio("Input mode", ["Text Area", "File Upload"], horizontal=True, key="input_mode")

input_text = ""
if input_mode == "Text Area":
    # Initialize text area state if empty
    if "text_area_widget" not in st.session_state:
        st.session_state["text_area_widget"] = ""
        
    input_text = st.text_area(
        "Enter or paste your text below:",
        height=200,
        placeholder="Type or paste text here, or select a sample from the sidebar...",
        key="text_area_widget"
    )
else:
    uploaded = st.file_uploader("Upload .txt or .md file", type=["txt", "md"])
    if uploaded:
        input_text = uploaded.read().decode("utf-8", errors="replace")
    else:
        st.info("Upload a text file to analyze.")

# Run pipeline
st.markdown("<br>", unsafe_allow_html=True)
if st.button("🚀 Analyze Document", type="primary", use_container_width=True) and input_text.strip():
    with st.spinner("Processing NLP Pipeline..."):
        try:
            result = run_pipeline(
                input_text,
                language=language,
                remove_stopwords_flag=remove_stopwords,
                lemmatize_flag=lemmatize,
            )

            if auto_detect_lang:
                detected = result.language
                flag = "🌍" if detected == "tigrinya" else "🇬🇧"
                st.success(f"{flag} Auto-detected language: **{detected.title()}**")

        except ImportError as e:
            st.error(f"Missing dependency: {e}")
            if "tigrinya" in str(e).lower():
                st.code("pip install tigrinya-nlp", language="bash")
            else:
                st.code("pip install nltk", language="bash")
            st.stop()
        except Exception as e:
            st.error(f"Error: {e}")
            st.exception(e)
            st.stop()

    # Language mismatch warning (only when manually overriding)
    if not auto_detect_lang:
        actual = _detect_language(input_text)
        if actual != language:
            st.warning(
                f"⚠️ **Language mismatch:** You selected **{language.title()}** manually, "
                f"but the text appears to be **{actual.title()}**. "
                "Enable Auto-detect for correct processing."
            )

    # Metrics
    st.subheader("📈 Overview")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Tokens", len(result.tokens))
    with col2:
        st.metric("After Stopwords", len(result.tokens_no_stopwords))
    with col3:
        st.metric("Unique Terms", len(result.frequency_table))
    with col4:
        st.metric("Lemmas", len(result.lemmas))

    # Token flow preview (expandable)
    with st.expander("🔍 Token Flow Preview (2.1 → 2.2 → 2.3)"):
        flow_col1, flow_col2, flow_col3 = st.columns(3)
        with flow_col1:
            st.markdown("**Tokens**")
            st.code(" → ".join(result.tokens[:30]) + (" ..." if len(result.tokens) > 30 else ""), language=None)
        with flow_col2:
            st.markdown("**After Stopwords (2.1)**")
            st.code(" → ".join(result.tokens_no_stopwords[:30]) + (" ..." if len(result.tokens_no_stopwords) > 30 else ""), language=None)
        with flow_col3:
            st.markdown("**Lemmas (2.2)**")
            st.code(" → ".join(result.lemmas[:30]) + (" ..." if len(result.lemmas) > 30 else ""), language=None)

    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📋 Frequencies (2.3)",
        "📍 Distances (2.4)",
        "⚖️ Compound Relevance (2.5)",
        "📊 Visualizations",
        "📄 Full Results",
    ])

    with tab1:
        st.markdown("#### Term Frequencies (Requirement 2.3)")
        if result.frequency_table:
            freq_df = pd.DataFrame(
                list(result.frequency_table.items()),
                columns=["Term", "Frequency"],
            ).sort_values("Frequency", ascending=False)
            st.dataframe(freq_df, use_container_width=True, hide_index=True)
            fig = px.bar(
                freq_df.head(20),
                x="Term",
                y="Frequency",
                title="Top 20 Terms by Frequency",
                color="Frequency",
                color_continuous_scale="Viridis",
            )
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e8e8e8"),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No terms to display.")

    with tab2:
        st.markdown("#### Distance from Strategic Points (Requirement 2.4)")
        st.caption("Distance from start = first_position / total_tokens | Distance from end = (total - last_position) / total")
        if result.term_stats:
            dist_df = pd.DataFrame([
                {
                    "Term": t.term,
                    "First Pos": t.first_position,
                    "Last Pos": t.last_position,
                    "Dist. from Start": t.distance_from_start,
                    "Dist. from End": t.distance_from_end,
                }
                for t in result.term_stats
            ])
            st.dataframe(dist_df, use_container_width=True, hide_index=True)
            fig = px.scatter(
                dist_df.head(30),
                x="Dist. from Start",
                y="Dist. from End",
                size="First Pos",
                hover_data=["Term"],
                title="Term Positions: Start vs End Distance",
            )
            fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No term statistics.")

    with tab3:
        st.markdown("#### Compound Relevance (Requirement 2.5)")
        st.caption("Formula: 0.5 × frequency_score + 0.5 × earliness_score")
        if result.term_stats:
            rel_df = pd.DataFrame([
                {
                    "Term": t.term,
                    "Frequency": t.frequency,
                    "Freq Score": t.freq_score,
                    "Earliness Score": t.earliness_score,
                    "Compound Relevance": t.compound_relevance,
                }
                for t in result.term_stats
            ])
            st.dataframe(rel_df, use_container_width=True, hide_index=True)
            fig = px.bar(
                rel_df.head(20),
                x="Term",
                y="Compound Relevance",
                color="Compound Relevance",
                color_continuous_scale="Plasma",
                title="Top 20 Terms by Compound Relevance",
            )
            fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No term statistics.")

    with tab4:
        st.markdown("#### Visualizations")
        if result.term_stats:
            rel_df = pd.DataFrame([
                {"Term": t.term, "Relevance": t.compound_relevance, "Freq Score": t.freq_score, "Earliness": t.earliness_score}
                for t in result.term_stats
            ])
            fig = go.Figure(data=[
                go.Bar(name="Compound Relevance", x=rel_df["Term"].head(15), y=rel_df["Relevance"].head(15)),
            ])
            fig.update_layout(
                title="Compound Relevance by Term",
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig, use_container_width=True)

            # Freq vs Earliness scatter (2.5 formula components)
            st.markdown("##### Frequency Score vs Earliness Score (2.5 formula components)")
            fig_scatter = px.scatter(
                rel_df.head(30),
                x="Freq Score",
                y="Earliness",
                size="Relevance",
                hover_data=["Term"],
                text="Term",
                title="Freq Score × Earliness → Compound Relevance",
            )
            fig_scatter.update_traces(textposition="top center", textfont_size=10)
            fig_scatter.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_scatter, use_container_width=True)
        if result.frequency_table:
            freq_df = pd.DataFrame(list(result.frequency_table.items()), columns=["Term", "Count"])
            fig2 = px.pie(freq_df.head(15), values="Count", names="Term", title="Term Distribution (Top 15)")
            fig2.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig2, use_container_width=True)

    with tab5:
        st.markdown("#### Full Pipeline Output")
        full_output = {
            "tokens": result.tokens[:50],
            "tokens_no_stopwords": result.tokens_no_stopwords[:50],
            "lemmas": result.lemmas[:50],
            "frequency_table": dict(list(result.frequency_table.items())[:20]),
            "term_stats": [
                {
                    "term": t.term,
                    "frequency": t.frequency,
                    "first_position": t.first_position,
                    "last_position": t.last_position,
                    "distance_from_start": t.distance_from_start,
                    "distance_from_end": t.distance_from_end,
                    "freq_score": t.freq_score,
                    "earliness_score": t.earliness_score,
                    "compound_relevance": t.compound_relevance,
                }
                for t in result.term_stats[:30]
            ],
        }
        st.json(full_output)

        # Export buttons
        export_col1, export_col2 = st.columns(2)
        with export_col1:
            csv_data = pd.DataFrame([
                {
                    "Term": t.term,
                    "Frequency": t.frequency,
                    "Dist_Start": t.distance_from_start,
                    "Dist_End": t.distance_from_end,
                    "Compound_Relevance": t.compound_relevance,
                }
                for t in result.term_stats
            ]).to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download CSV", csv_data, "nlp_results.csv", "text/csv", use_container_width=True)
        with export_col2:
            json_data = json.dumps(full_output, indent=2, ensure_ascii=False).encode("utf-8")
            st.download_button("📥 Download JSON", json_data, "nlp_results.json", "application/json", use_container_width=True)

else:
    if not input_text.strip():
        st.info("👆 Enter text and click **Analyze** to run the pipeline.")

st.divider()
st.caption("NLP Graphic Tool — Research Assignment Option 2 | Tigrinya & English | Streamlit")
