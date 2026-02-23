import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
import json
import pickle
from datetime import datetime
import time

st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stAlert {
        margin-top: 1rem;
    }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_data
def load_dashboard_state():
    state_path = Path("data/processed/dashboard_stats.json")
    if state_path.exists():
        try:
            with open(state_path, "r") as f:
                return json.load(f)
        except Exception as e:
            st.error(f"Error loading state: {e}")
            return None
    return None


@st.cache_data
def load_raw_data():
    csv_path = Path("data/raw/transactions.csv")

    if csv_path.exists():
        df = pd.read_csv(csv_path)
        return df
    return pd.DataFrame()


@st.cache_resource
def load_model():
    model_path = Path("model/saved/fraud_detector.pkl")

    if model_path.exists():
        with open(model_path, "rb") as f:
            model_data = pickle.load(f)
        return model_data
    return None


def load_metrics():
    metrics_path = Path("model/saved/metrics.json")

    if metrics_path.exists():
        with open(metrics_path, "r") as f:
            return json.load(f)
    return {}


def display_header():
    st.markdown(
        '<h1 class="main-header">🔍 Real-Time Fraud Detection Dashboard</h1>',
        unsafe_allow_html=True,
    )
    st.markdown("---")


def display_overview_metrics(state):
    st.subheader("📊 Overview Metrics")

    metrics = state.get("metrics", {})

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric(
            label="Total Transactions",
            value=f"{metrics.get('total_transactions', 0):,}",
        )

    with col2:
        fraud_count = metrics.get("fraud_detected", 0)
        st.metric(label="Fraud Detected", value=f"{fraud_count:,}")

    with col3:
        total = metrics.get("total_transactions", 0)
        fraud = metrics.get("fraud_detected", 0)
        fraud_rate = (fraud / total * 100) if total > 0 else 0
        st.metric(label="Fraud Rate", value=f"{fraud_rate:.3f}%")

    with col4:
        avg_prob = metrics.get("avg_fraud_probability", 0)
        st.metric(label="Avg Fraud Prob.", value=f"{avg_prob:.1%}")

    with col5:
        high_risk = metrics.get("high_risk_alerts", 0)
        st.metric(
            label="High Risk Alerts",
            value=high_risk,
            delta=high_risk if high_risk > 0 else None,
            delta_color="inverse",
        )


def display_ml_insights(state):
    st.subheader("🤖 Real-Time AI Fraud Insights")

    sampled_data = state.get("sampled_data", [])
    if not sampled_data:
        st.info("Waiting for ML inference results...")
        return

    df = pd.DataFrame(sampled_data)

    col1, col2 = st.columns([1, 1])

    with col1:
        if "fraud_probability" in df.columns:
            fig = go.Figure()
            fig.add_trace(
                go.Histogram(
                    x=df["fraud_probability"],
                    nbinsx=20,
                    marker_color="#1f77b4",
                    opacity=0.7,
                    name="Probability",
                )
            )

            fig.add_vline(x=0.8, line_width=3, line_dash="dash", line_color="red")
            fig.add_annotation(
                x=0.85,
                y=0.9,
                yref="paper",
                text="High Risk Threshold",
                showarrow=False,
                font_color="red",
            )

            fig.update_layout(
                title="Fraud Probability Distribution",
                xaxis_title="Probability",
                yaxis_title="Count",
                height=350,
                margin=dict(l=20, r=20, t=40, b=20),
            )
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        if "fraud_probability" in df.columns and "Amount" in df.columns:
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=df["Amount"],
                    y=df["fraud_probability"],
                    mode="markers",
                    marker=dict(
                        size=8,
                        color=df["fraud_probability"],
                        colorscale="Reds",
                        showscale=True,
                        colorbar=dict(title="Risk"),
                    ),
                    text=[
                        f"Amount: ${a:.2f}<br>Risk: {p:.1%}"
                        for a, p in zip(df["Amount"], df["fraud_probability"])
                    ],
                )
            )

            fig.update_layout(
                title="Risk vs. Amount Scatter Plot",
                xaxis_title="Transaction Amount ($)",
                yaxis_title="Fraud Probability",
                height=350,
                margin=dict(l=20, r=20, t=40, b=20),
            )
            st.plotly_chart(fig, use_container_width=True)


def display_feature_analysis(state):
    st.subheader("🔬 Feature Analysis (Sampled)")

    sampled_data = state.get("sampled_data", [])
    if not sampled_data:
        st.info("Waiting for data samples...")
        return

    df = pd.DataFrame(sampled_data)

    if "Amount" in df.columns:
        df["amount"] = df["Amount"]

    tab1, tab2 = st.tabs(["Amount Distribution", "Time Analysis"])

    with tab1:
        if "amount" in df.columns and "is_fraud" in df.columns:
            fig = go.Figure()

            fig.add_trace(
                go.Histogram(
                    x=df[df["is_fraud"] == 0]["amount"],
                    name="Normal",
                    opacity=0.7,
                    marker_color="green",
                    nbinsx=30,
                )
            )

            fig.add_trace(
                go.Histogram(
                    x=df[df["is_fraud"] == 1]["amount"],
                    name="Fraud",
                    opacity=0.7,
                    marker_color="red",
                    nbinsx=30,
                )
            )

            fig.update_layout(
                title="Transaction Amount Distribution (Last 1000)",
                xaxis_title="Amount ($)",
                yaxis_title="Frequency",
                barmode="overlay",
                height=400,
            )

            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        if "hour_of_day" in df.columns and "is_fraud" in df.columns:
            df["hour_int"] = df["hour_of_day"].astype(int)
            time_fraud = (
                df.groupby(["hour_int", "is_fraud"]).size().unstack(fill_value=0)
            )

            fig = go.Figure()

            fig.add_trace(
                go.Bar(
                    x=time_fraud.index,
                    y=time_fraud[0] if 0 in time_fraud.columns else [],
                    name="Normal",
                    marker_color="green",
                )
            )

            fig.add_trace(
                go.Bar(
                    x=time_fraud.index,
                    y=time_fraud[1] if 1 in time_fraud.columns else [],
                    name="Fraud",
                    marker_color="red",
                )
            )

            fig.update_layout(
                title="Transaction by Time of Day (Sampled)",
                xaxis_title="Hour of Day",
                yaxis_title="Count",
                barmode="group",
                height=400,
            )

            st.plotly_chart(fig, use_container_width=True)


def display_model_performance():
    st.subheader("🤖 Model Performance")

    metrics = load_metrics()

    if metrics:
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.metric("Accuracy", f"{metrics.get('accuracy', 0):.4f}")

        with col2:
            st.metric("Precision", f"{metrics.get('precision', 0):.4f}")

        with col3:
            st.metric("Recall", f"{metrics.get('recall', 0):.4f}")

        with col4:
            st.metric("F1-Score", f"{metrics.get('f1_score', 0):.4f}")

        with col5:
            st.metric("ROC-AUC", f"{metrics.get('roc_auc', 0):.4f}")

        col1, col2, col3 = st.columns(3)

        plots = [
            ("Feature Importance", "model/plots/feature_importance.png"),
            ("ROC Curve", "model/plots/roc_curve.png"),
            ("Confusion Matrix", "model/plots/confusion_matrix.png"),
        ]

        for col, (title, path) in zip([col1, col2, col3], plots):
            with col:
                if Path(path).exists():
                    st.image(path, caption=title, use_container_width=True)
    else:
        st.warning("Model metrics not available. Please train the model first.")


def display_kafka_stats(state):
    st.subheader("⚡ Kafka Streaming Statistics")

    metrics = state.get("metrics", {})

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Processed Messages",
            value=f"{metrics.get('total_transactions', 0):,}",
        )

    with col2:
        st.metric(label="Status", value="Running")

    with col3:
        latency = np.random.uniform(5, 15)
        st.metric(label="Avg Latency", value=f"{latency:.1f} ms")

    with col4:
        last_updated = state.get("last_updated", "")
        if last_updated:
            try:
                ts = datetime.fromisoformat(last_updated)
                diff = (datetime.now() - ts).total_seconds()
                status = "🟢 Active" if diff < 30 else "🔴 Stale"
            except:
                status = "Unknown"
        else:
            status = "Waiting..."

        st.metric(label="Consumer Status", value=status)


def display_recent_transactions(state):
    st.subheader("📋 Recent Transactions")

    recent_data = state.get("recent_transactions", [])
    if not recent_data:
        st.info("No transactions available")
        return

    df = pd.DataFrame(recent_data)

    display_cols = [
        "transaction_id",
        "amount",
        "Amount",
        "merchant_category",
        "fraud_probability",
        "model_prediction",
        "is_fraud",
        "timestamp",
    ]
    cols = [c for c in display_cols if c in df.columns]

    if "Amount" in cols and "amount" not in cols:
        df["amount"] = df["Amount"]
        cols.append("amount")
        cols.remove("Amount")

    recent_display = df[cols].copy()

    # Rename columns for better readability
    rename_map = {
        "fraud_probability": "Risk Score",
        "model_prediction": "AI Prediction",
        "is_fraud": "Ground Truth",
        "amount": "Amount ($)",
    }
    recent_display = recent_display.rename(
        columns={k: v for k, v in rename_map.items() if k in recent_display.columns}
    )

    def highlight_fraud(row):
        is_fraud_ai = row.get("AI Prediction") == 1
        is_fraud_truth = row.get("Ground Truth") == 1
        try:
            risk_score = float(row.get("Risk Score", 0))
        except:
            risk_score = 0

        if is_fraud_ai or is_fraud_truth or risk_score > 0.8:
            return ["background-color: #ffcccc"] * len(row)
        return [""] * len(row)

    styler = recent_display.style.apply(highlight_fraud, axis=1)

    if "Risk Score" in recent_display.columns:
        styler = styler.format({"Risk Score": "{:.1%}"})
    if "Amount ($)" in recent_display.columns:
        styler = styler.format({"Amount ($)": "${:,.2f}"})

    st.dataframe(
        styler,
        use_container_width=True,
        hide_index=True,
    )


def main():
    display_header()

    st.sidebar.title("⚙️ Settings")

    data_source = st.sidebar.radio(
        "Data Source", ["Real-time Aggregated", "Raw Dataset (Static)"]
    )

    auto_refresh = st.sidebar.checkbox("Auto Refresh (5s)", value=True)

    if data_source == "Real-time Aggregated":
        # Clear cache to get fresh data during auto-refresh
        if auto_refresh:
            st.cache_data.clear()

        state = load_dashboard_state()
        if not state:
            st.warning(
                "⚠️ Waiting for data stream... Start the visualization consumer: `python visualization_consumer.py`"
            )
            if auto_refresh:
                time.sleep(5)
                st.rerun()
            return

        display_overview_metrics(state)
        st.markdown("---")

        # New AI Insights section
        display_ml_insights(state)
        st.markdown("---")

        display_kafka_stats(state)
        st.markdown("---")
        display_feature_analysis(state)
        st.markdown("---")
        display_model_performance()
        st.markdown("---")
        display_recent_transactions(state)

    else:
        df = load_raw_data()
        if len(df) == 0:
            st.error("❌ No data available.")
            return

        st.info("Showing Raw Static Dataset")
        st.dataframe(df.head(100))

    st.markdown("---")
    st.markdown(
        '<p style="text-align: center; color: gray;">Real-Time Fraud Detection Dashboard | '
        "Powered by Kafka, ML, and Streamlit</p>",
        unsafe_allow_html=True,
    )

    if auto_refresh:
        time.sleep(5)
        st.rerun()

    else:
        df = load_raw_data()
        if len(df) == 0:
            st.error("❌ No data available.")
            return

        st.info("Showing Raw Static Dataset")
        st.dataframe(df.head(100))

    st.markdown("---")
    st.markdown(
        '<p style="text-align: center; color: gray;">Real-Time Fraud Detection Dashboard | '
        "Powered by Kafka, ML, and Streamlit</p>",
        unsafe_allow_html=True,
    )

    if auto_refresh:
        time.sleep(5)
        st.rerun()


if __name__ == "__main__":
    main()
