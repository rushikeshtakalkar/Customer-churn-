import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

st.set_page_config(page_title="Customer Churn & Retention", page_icon="📊", layout="wide")

DATA_PATH = Path(__file__).parent / "customer_churn_synthetic_dataset.csv"

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    return df

df = load_data()
st.title("📊 Customer Churn & Retention Analysis")
st.caption("Portfolio demo • Dataset is synthetic for demonstration; replace with a verified public dataset before presenting findings as real-world results.")

# Normalize common expected column names without silently fabricating data
lower_map = {c.lower().strip().replace(" ", "_"): c for c in df.columns}
def find_col(*names):
    for name in names:
        if name in lower_map:
            return lower_map[name]
    return None

churn_col = find_col("churn", "churn_label", "exited")
tenure_col = find_col("tenure", "tenure_months")
contract_col = find_col("contract", "contract_type")
monthly_col = find_col("monthlycharges", "monthly_charges", "monthly_charge")
if monthly_col is None:
    monthly_col = find_col("monthlycharges")
customer_col = find_col("customerid", "customer_id", "id")

# Standardize churn values to readable strings
if churn_col:
    df["_churn_display"] = df[churn_col].astype(str).str.strip()
else:
    st.error("The dataset does not contain a recognizable churn column.")
    st.stop()

total = len(df)
churn_norm = df["_churn_display"].str.lower()
churned_mask = churn_norm.isin(["yes", "true", "1", "churned", "exited"])
churned = int(churned_mask.sum())
churn_rate = (churned / total * 100) if total else 0
retained = total - churned

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Customers", f"{total:,}")
c2.metric("Churned Customers", f"{churned:,}")
c3.metric("Churn Rate", f"{churn_rate:.1f}%")
c4.metric("Retained Customers", f"{retained:,}")

st.divider()
left, right = st.columns(2)

with left:
    st.subheader("Churn Distribution")
    counts = df["_churn_display"].value_counts()
    fig, ax = plt.subplots()
    counts.plot(kind="bar", ax=ax)
    ax.set_xlabel("Churn status")
    ax.set_ylabel("Customer count")
    ax.set_title("Customers by Churn Status")
    plt.xticks(rotation=0)
    st.pyplot(fig, clear_figure=True)

with right:
    st.subheader("Churn by Segment")
    segment_options = []
    if contract_col:
        segment_options.append(contract_col)
    for c in df.columns:
        if c != churn_col and (df[c].dtype == "object" or str(df[c].dtype) == "category"):
            if c not in segment_options and df[c].nunique(dropna=True) <= 15:
                segment_options.append(c)
    if segment_options:
        segment = st.selectbox("Choose a customer segment", segment_options)
        segment_df = df.copy()
        segment_df["_is_churned"] = churned_mask.astype(int)
        summary = segment_df.groupby(segment, dropna=False)["_is_churned"].agg(["mean", "count"]).reset_index()
        summary["Churn Rate (%)"] = summary["mean"] * 100
        fig2, ax2 = plt.subplots()
        summary.set_index(segment)["Churn Rate (%)"].plot(kind="bar", ax=ax2)
        ax2.set_ylabel("Churn rate (%)")
        ax2.set_title(f"Churn Rate by {segment}")
        plt.xticks(rotation=35, ha="right")
        st.pyplot(fig2, clear_figure=True)
        st.dataframe(summary[[segment, "count", "Churn Rate (%)"]].rename(columns={"count":"Customers"}), use_container_width=True)
    else:
        st.info("No categorical segment columns found for segment analysis.")

if tenure_col:
    st.subheader("Tenure Analysis")
    tenure_num = pd.to_numeric(df[tenure_col], errors="coerce")
    temp = pd.DataFrame({"Tenure": tenure_num, "Churned": churned_mask.astype(int)}).dropna()
    if not temp.empty:
        temp["Tenure Band"] = pd.cut(temp["Tenure"], bins=[-1, 6, 12, 24, 48, 1200], labels=["0–6", "7–12", "13–24", "25–48", "49+"])
        tenure_summary = temp.groupby("Tenure Band", observed=False)["Churned"].agg(["mean", "count"]).reset_index()
        tenure_summary["Churn Rate (%)"] = tenure_summary["mean"] * 100
        fig3, ax3 = plt.subplots()
        tenure_summary.set_index("Tenure Band")["Churn Rate (%)"].plot(kind="bar", ax=ax3)
        ax3.set_ylabel("Churn rate (%)")
        ax3.set_title("Churn Rate by Tenure Band")
        plt.xticks(rotation=0)
        st.pyplot(fig3, clear_figure=True)

st.subheader("Explore Dataset")
search = st.text_input("Filter rows (search across displayed values)", "")
view_df = df.drop(columns=["_churn_display"], errors="ignore")
if search:
    mask = view_df.astype(str).apply(lambda col: col.str.contains(search, case=False, na=False)).any(axis=1)
    view_df = view_df[mask]
st.dataframe(view_df.head(500), use_container_width=True)
st.download_button("Download dataset CSV", data=df.to_csv(index=False).encode("utf-8"), file_name="customer_churn_dataset.csv", mime="text/csv")

with st.expander("Methodology & limitations"):
    st.markdown("""
    - KPIs are calculated directly from the included dataset.
    - Churn is counted from values such as Yes/True/1/Churned/Exited.
    - This starter app uses a **synthetic demonstration dataset**. Do not describe its metrics as findings about actual customers.
    - For a portfolio-ready analysis, replace the file with a public telecom churn dataset, validate column mappings, and document the real findings.
    """)
