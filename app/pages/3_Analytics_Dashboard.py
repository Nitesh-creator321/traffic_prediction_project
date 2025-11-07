import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ==========================================
# 📊 Model Analytics Dashboard
# ==========================================
st.set_page_config(page_title="📊 Model Analytics Dashboard", layout="wide")

st.title("📊 Model Analytics Dashboard")

st.markdown("""
This dashboard shows how different **Machine Learning (ML)** and **Deep Learning (DL)** models performed during training.  
You can compare their **accuracy, error metrics, and R² scores** to find the best traffic prediction model.
""")

# === Load previous ML results ===
try:
    results_df = pd.read_csv("model_results.csv")
    st.success("✅ Machine Learning model results loaded successfully.")
except FileNotFoundError:
    st.warning("⚠️ Model results file not found. Please run `main.py` first to train ML models.")
    st.stop()

# === Merge Deep Learning results if available ===
try:
    dl_df = pd.read_csv("dl_results.csv")
    results_df = pd.concat([results_df, dl_df], ignore_index=True)
    st.success("🧠 Deep Learning (LSTM) results added to analytics.")
except FileNotFoundError:
    st.warning("⚠️ Deep Learning results not found. Run `src/train_dl.py` or `main.py` to generate them.")

# === Identify best model ===
best_model_name = results_df.loc[results_df['R2'].idxmax(), 'Model']
best_r2 = results_df['R2'].max()
st.markdown(f"🏆 **Best Performing Model:** `{best_model_name}` with **R² = {best_r2:.4f}**")

# === Show raw table ===
st.subheader("📋 Model Performance Summary")
st.dataframe(results_df.style.highlight_max(axis=0, color="lightgreen"))

# === Plot R² comparison ===
st.subheader("📈 R² Score Comparison")

fig, ax = plt.subplots(figsize=(8, 4))
sns.barplot(x="Model", y="R2", data=results_df, ax=ax)
ax.set_title("Model R² Scores (Higher is Better)", fontsize=14)
ax.set_ylabel("R² Score")
st.pyplot(fig)

# === Plot MAE and MSE ===
col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots(figsize=(6, 3))
    sns.barplot(x="Model", y="MAE", data=results_df, ax=ax)
    ax.set_title("Mean Absolute Error (MAE)")
    ax.set_ylabel("Error Value")
    ax.set_xlabel("Model")
    st.pyplot(fig)

with col2:
    fig, ax = plt.subplots(figsize=(6, 3))
    sns.barplot(x="Model", y="MSE", data=results_df, ax=ax)
    ax.set_title("Mean Squared Error (MSE)")
    ax.set_ylabel("Error Value")
    ax.set_xlabel("Model")
    st.pyplot(fig)

# === Summary Stats ===
st.markdown("---")
st.header("📊 Summary Insights")

col1, col2, col3 = st.columns(3)
col1.metric("📉 Lowest MAE", f"{results_df['MAE'].min():.2f}")
col2.metric("📉 Lowest MSE", f"{results_df['MSE'].min():.2f}")
col3.metric("📈 Highest R²", f"{results_df['R2'].max():.3f}")

# === Highlight Deep Learning Section ===
if 'LSTM' in results_df['Model'].values:
    st.markdown("""
    ---
    ### 🧠 Deep Learning Model Insights (LSTM)
    The **LSTM (Long Short-Term Memory)** model captures sequential traffic patterns over time.  
    It’s designed to handle temporal dependencies — making it more accurate during high traffic fluctuation hours.
    """)
    lstm_row = results_df[results_df["Model"] == "LSTM"].iloc[0]
    st.markdown(f"""
    - **MAE:** {lstm_row["MAE"]:.2f}  
    - **MSE:** {lstm_row["MSE"]:.2f}  
    - **R²:** {lstm_row["R2"]:.4f}  
    """)

# === Footer ===
st.markdown("<hr><center>Developed by <b>Nitesh & Team 🚀 | BMSIT</b></center>", unsafe_allow_html=True)
