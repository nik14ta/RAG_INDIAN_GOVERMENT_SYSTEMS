import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

st.set_page_config(page_title="RAG Evaluation Dashboard", page_icon="📊", layout="wide")

RESULTS_PATH = "results/experiment_results.csv"

st.title("📊 RAG Evaluation Dashboard")
st.caption("Comparing Baseline LLM vs Advanced Hybrid RAG on Hallucination Reduction, Faithfulness & Abstention.")

if not os.path.exists(RESULTS_PATH):
    st.warning("No results found. Please run: `cd evaluation && python run_experiments.py` first.")
    st.stop()

df = pd.read_csv(RESULTS_PATH)
total = len(df)
answerable = df[df['Is_Answerable'] == True]
unanswerable = df[df['Is_Answerable'] == False]

# ---- CALCULATE METRICS ----
rag_abstention_acc    = (df['RAG_Correct_Abstention'].sum() / total) * 100
base_abstention_acc   = (df['Baseline_Correct_Abstention'].sum() / total) * 100

# Hallucination = unanswerable questions where model did NOT abstain
rag_hallucinated      = unanswerable[~unanswerable['RAG_Correct_Abstention']] if len(unanswerable) else pd.DataFrame()
base_hallucinated     = unanswerable[~unanswerable['Baseline_Correct_Abstention']] if len(unanswerable) else pd.DataFrame()
rag_hallucination_pct = (len(rag_hallucinated) / len(unanswerable) * 100) if len(unanswerable) else 0
base_hallucination_pct= (len(base_hallucinated) / len(unanswerable) * 100) if len(unanswerable) else 0

# Faithfulness ≈ answerable questions where RAG did NOT abstain (answered when it should have)
rag_faithful          = answerable[answerable['RAG_Correct_Abstention']] if len(answerable) else pd.DataFrame()
base_faithful         = answerable[answerable['Baseline_Correct_Abstention']] if len(answerable) else pd.DataFrame()
rag_faithfulness_pct  = (len(rag_faithful) / len(answerable) * 100) if len(answerable) else 0
base_faithfulness_pct = (len(base_faithful) / len(answerable) * 100) if len(answerable) else 0

# ---- HIGH-LEVEL METRIC CARDS ----
st.markdown("### 🏆 Key Metrics")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Questions", total)
c2.metric("RAG Abstention Accuracy",  f"{rag_abstention_acc:.1f}%",  f"+{rag_abstention_acc - base_abstention_acc:.1f}% vs Baseline")
c3.metric("RAG Faithfulness",         f"{rag_faithfulness_pct:.1f}%", f"+{rag_faithfulness_pct - base_faithfulness_pct:.1f}% vs Baseline")
c4.metric("RAG Hallucination Rate",   f"{rag_hallucination_pct:.1f}%", f"{rag_hallucination_pct - base_hallucination_pct:.1f}% vs Baseline", delta_color="inverse")

st.markdown("---")

# ---- RAG vs NON-RAG BAR CHARTS ----
st.markdown("### 📈 RAG vs Non-RAG Comparison")

col_left, col_right = st.columns(2)

with col_left:
    st.markdown("**Abstention Accuracy (%)**")
    fig, ax = plt.subplots(figsize=(5, 3))
    bars = ax.barh(['Non-RAG Baseline', 'Hybrid RAG'], [base_abstention_acc, rag_abstention_acc],
                   color=['#ff9999', '#66b3ff'])
    ax.set_xlim(0, 100)
    for bar in bars:
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                f"{bar.get_width():.1f}%", va='center', fontweight='bold')
    ax.set_xlabel('Accuracy (%)')
    st.pyplot(fig)

with col_right:
    st.markdown("**Hallucination Rate (%) — Lower is Better**")
    fig2, ax2 = plt.subplots(figsize=(5, 3))
    bars2 = ax2.barh(['Non-RAG Baseline', 'Hybrid RAG'], [base_hallucination_pct, rag_hallucination_pct],
                     color=['#ff9999', '#66b3ff'])
    ax2.set_xlim(0, 100)
    for bar in bars2:
        ax2.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                 f"{bar.get_width():.1f}%", va='center', fontweight='bold')
    ax2.set_xlabel('Hallucination Rate (%)')
    st.pyplot(fig2)

st.markdown("---")

# ---- FAITHFULNESS TABLE ----
st.markdown("### 📋 Faithfulness & Retrieval")

f_col1, f_col2 = st.columns(2)
with f_col1:
    st.markdown("**Faithfulness (Answerable Questions Handled Correctly)**")
    faith_df = pd.DataFrame({
        "System": ["Non-RAG Baseline", "Hybrid RAG"],
        "Faithfulness (%)": [round(base_faithfulness_pct, 1), round(rag_faithfulness_pct, 1)]
    })
    st.dataframe(faith_df, use_container_width=True, hide_index=True)

with f_col2:
    st.markdown("**Hallucination Breakdown**")
    hall_df = pd.DataFrame({
        "System": ["Non-RAG Baseline", "Hybrid RAG"],
        "Hallucinated Answers": [len(base_hallucinated), len(rag_hallucinated)],
        "Total Unanswerable Qs": [len(unanswerable), len(unanswerable)]
    })
    st.dataframe(hall_df, use_container_width=True, hide_index=True)

st.markdown("---")

# ---- DETAILED QUERY TABLE ----
st.markdown("### 🔍 Full Query Analysis")
st.dataframe(
    df[['Question', 'Is_Answerable', 'Baseline_Answer', 'RAG_Answer',
        'RAG_Correct_Abstention', 'Baseline_Correct_Abstention']],
    use_container_width=True
)
