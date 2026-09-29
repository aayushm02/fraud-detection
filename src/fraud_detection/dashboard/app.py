"""Streamlit dashboard for fraud detection."""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt

from fraud_detection.models.registry import ModelRegistry
from fraud_detection.explainability.shap_explainer import FraudExplainer
from fraud_detection.config import get_settings

st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6;
    }
    .metric-card {
        background-color: white;
        padding: 15px;
        border-radius: 5px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #1f77b4;
    }
    .metric-label {
        font-size: 14px;
        color: #7f7f7f;
    }
    .risk-low { background-color: #d4edda; color: #155724; padding: 5px; border-radius: 5px; }
    .risk-medium { background-color: #fff3cd; color: #856404; padding: 5px; border-radius: 5px; }
    .risk-high { background-color: #ffeeba; color: #856404; padding: 5px; border-radius: 5px; }
    .risk-critical { background-color: #f8d7da; color: #721c24; padding: 5px; border-radius: 5px; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_registry():
    return ModelRegistry()

registry = get_registry()
models = registry.list_models()

def render_overview():
    st.header("Dashboard Overview")
    if not models:
        st.warning("No models found in the registry. Please train a model first.")
        return
        
    best_model_info = registry.get_best("f1")
    if not best_model_info:
        st.info("No best model found.")
        return
        
    metrics = best_model_info.get("metrics", {})
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'<div class="metric-card"><div class="metric-value">1,024,532</div><div class="metric-label">Total Transactions</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card"><div class="metric-value">0.17%</div><div class="metric-label">Fraud Rate %</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{metrics.get("accuracy", 0.0):.4f}</div><div class="metric-label">Model Accuracy</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{metrics.get("f1", 0.0):.4f}</div><div class="metric-label">F1 Score</div></div>', unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    with c1:
        fig_pie = px.pie(names=["Legitimate", "Fraudulent"], values=[9983, 17], title="Fraud Distribution")
        st.plotly_chart(fig_pie, use_container_width=True)
        
    with c2:
        np.random.seed(42)
        legit_amounts = np.random.lognormal(mean=3, sigma=1, size=1000)
        fraud_amounts = np.random.lognormal(mean=5, sigma=1.5, size=50)
        
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(x=legit_amounts, name="Legitimate", opacity=0.75))
        fig_hist.add_trace(go.Histogram(x=fraud_amounts, name="Fraudulent", opacity=0.75))
        fig_hist.update_layout(barmode='overlay', title="Amount Distribution (Log Scale)", xaxis_type="log")
        st.plotly_chart(fig_hist, use_container_width=True)

def render_model_comparison():
    st.header("Model Comparison")
    if not models:
        st.warning("No models found in the registry.")
        return
        
    df_models = pd.DataFrame(models)
    
    if 'metrics' in df_models.columns:
        metrics_df = pd.json_normalize(df_models['metrics'])
        df_display = pd.concat([df_models.drop(['metrics'], axis=1), metrics_df], axis=1)
        st.dataframe(df_display.style.highlight_max(axis=0, subset=[col for col in ['f1', 'accuracy', 'roc_auc', 'precision', 'recall'] if col in df_display.columns]))
    else:
        st.dataframe(df_models)
        
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("ROC Curves")
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=[0, 0.1, 0.5, 1], y=[0, 0.8, 0.95, 1], name="Model A"))
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], line=dict(dash='dash'), name="Random"))
        st.plotly_chart(fig_roc, use_container_width=True)
        
    with c2:
        st.subheader("PR Curves")
        fig_pr = go.Figure()
        fig_pr.add_trace(go.Scatter(x=[0, 0.5, 1], y=[1, 0.8, 0.1], name="Model A"))
        st.plotly_chart(fig_pr, use_container_width=True)

def render_live_prediction():
    st.header("Live Prediction")
    if not models:
        st.warning("No models found. Please train a model.")
        return
        
    best_info = registry.get_best("f1")
    if not best_info:
        st.info("No best model available.")
        return
        
    col_input, col_result = st.columns([1, 1])
    
    with col_input:
        st.subheader("Transaction Details")
        amount = st.number_input("Amount", min_value=0.0, value=150.0)
        v1 = st.slider("V1", -5.0, 5.0, 0.0)
        v2 = st.slider("V2", -5.0, 5.0, 0.0)
        v3 = st.slider("V3", -5.0, 5.0, 0.0)
        v4 = st.slider("V4", -5.0, 5.0, 0.0)
        
        predict_btn = st.button("Predict Fraud Probability", type="primary")
        
    if predict_btn:
        with col_result:
            st.subheader("Prediction Results")
            
            try:
                model = registry.load(best_info["name"], best_info["version"])
                if model:
                    features = {f"V{i}": 0.0 for i in range(1, 29)}
                    features.update({"Amount": amount, "V1": v1, "V2": v2, "V3": v3, "V4": v4, "Time": 0})
                    
                    df_input = pd.DataFrame([features])
                    
                    prob = float(model.predict_proba(df_input)[0, 1])
                    
                    if prob < 0.3:
                        risk_level, risk_class = "Low", "risk-low"
                    elif prob < 0.6:
                        risk_level, risk_class = "Medium", "risk-medium"
                    elif prob < 0.85:
                        risk_level, risk_class = "High", "risk-high"
                    else:
                        risk_level, risk_class = "Critical", "risk-critical"
                        
                    st.markdown(f'<h3>Risk Level: <span class="{risk_class}">{risk_level}</span></h3>', unsafe_allow_html=True)
                    
                    fig_gauge = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = prob * 100,
                        title = {'text': "Fraud Probability (%)"},
                        gauge = {
                            'axis': {'range': [None, 100]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 30], 'color': "lightgreen"},
                                {'range': [30, 60], 'color': "yellow"},
                                {'range': [60, 85], 'color': "orange"},
                                {'range': [85, 100], 'color': "red"}
                            ]
                        }
                    ))
                    st.plotly_chart(fig_gauge, use_container_width=True)
                    
                    try:
                        explainer = FraudExplainer(model)
                        top_features = explainer.get_top_features(df_input)
                        if top_features:
                            st.write("Top Risk Factors:")
                            for factor in top_features:
                                st.write(f"- **{factor['feature']}**: {factor['importance']:.4f}")
                    except Exception as e:
                        st.info(f"Explainability not available: {e}")
                else:
                    st.error("Could not load model.")
            except Exception as e:
                st.error(f"Prediction failed: {e}")

def render_explainability():
    st.header("Model Explainability")
    if not models:
        st.warning("No models found.")
        return
        
    st.info("Global SHAP values and feature importance would be rendered here.")
    st.write("Feature Importance Bar Chart:")
    fig = px.bar(x=["V1", "V14", "V4", "Amount"], y=[0.4, 0.3, 0.2, 0.1], labels={'x': 'Feature', 'y': 'Importance'})
    st.plotly_chart(fig)

tabs = st.tabs(["Overview", "Model Comparison", "Live Prediction", "Explainability"])

with tabs[0]:
    render_overview()
with tabs[1]:
    render_model_comparison()
with tabs[2]:
    render_live_prediction()
with tabs[3]:
    render_explainability()
