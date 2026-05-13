"""
app.py — CLV Forecaster Streamlit Dashboard
=============================================
Professional 6-tab dashboard for CLV analysis.

Run:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ── Page Config ──────────────────────────────────────────
st.set_page_config(
    page_title="CLV Forecaster — Customer Lifetime Value",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .main { background: linear-gradient(135deg, #0f0c29, #1a1a2e, #16213e); }
    .stApp { background: linear-gradient(135deg, #0f0c29, #1a1a2e, #16213e); }
    .metric-card {
        background: linear-gradient(145deg, rgba(26,26,46,0.9), rgba(22,33,62,0.9));
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px; padding: 20px 24px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        text-align: center; transition: transform 0.2s, box-shadow 0.2s;
    }
    .metric-card:hover { transform: translateY(-4px); box-shadow: 0 12px 40px rgba(0,0,0,0.5); }
    .metric-value { font-size: 2rem; font-weight: 700; }
    .metric-label { font-size: 0.85rem; color: #8892b0; margin-top: 4px; }
    .tier-badge {
        display: inline-block; padding: 6px 16px; border-radius: 20px;
        font-weight: 600; font-size: 0.9rem; margin: 4px;
    }
    .header-gradient {
        background: linear-gradient(90deg, #667eea, #764ba2);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        font-size: 2.2rem; font-weight: 700;
    }
    div[data-testid="stSidebar"] { background: linear-gradient(180deg, #0f0c29, #1a1a2e); }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background: rgba(255,255,255,0.05); border-radius: 8px;
        padding: 8px 20px; font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea, #764ba2) !important;
    }
    .strategy-box {
        background: rgba(102,126,234,0.1); border-left: 4px solid #667eea;
        padding: 16px 20px; border-radius: 0 12px 12px 0; margin: 12px 0;
    }
</style>
""", unsafe_allow_html=True)


# ── Helper: Auto-bootstrap ───────────────────────────────
@st.cache_resource
def bootstrap_pipeline():
    """Generate data and train models if not already done."""
    if not os.path.exists("data/customers.csv"):
        with st.spinner("Generating synthetic dataset..."):
            import subprocess, sys
            subprocess.run([sys.executable, "generate_data.py"], check=True)
    if not os.path.exists("models/rf_model.pkl"):
        with st.spinner("Training ML models (this takes ~30s)..."):
            import subprocess, sys
            subprocess.run([sys.executable, "train_model.py"], check=True)


@st.cache_resource
def load_models():
    """Load all trained models and artifacts."""
    bootstrap_pipeline()
    return {
        'rf': joblib.load("models/rf_model.pkl"),
        'ridge': joblib.load("models/ridge_model.pkl"),
        'xgb': joblib.load("models/xgb_model.pkl"),
        'scaler': joblib.load("models/scaler.pkl"),
        'shap': joblib.load("models/shap_explainer.pkl"),
        'metrics': joblib.load("models/metrics.pkl"),
    }


@st.cache_data
def load_data():
    """Load scored customer dataset."""
    bootstrap_pipeline()
    return pd.read_csv("data/customers_clv.csv")


# ── Load everything ──────────────────────────────────────
try:
    models = load_models()
    df = load_data()
    DATA_READY = True
except Exception as e:
    DATA_READY = False
    st.error(f"Failed to load data/models: {e}")
    st.info("Run `python generate_data.py` then `python train_model.py` first.")
    st.stop()

# Import forecaster functions
from forecaster import (
    FEATURES, CLV_TIERS, FEATURE_META, predict_clv, get_shap_values,
    simulate_clv_change, get_summary_stats, get_clv_scatter,
    get_rfm_clv_heatmap, get_model_comparison_chart, get_shap_bar_chart,
    get_clv_distribution, get_feature_importance_chart,
    get_tier_revenue_chart, get_cohort_clv_chart, get_actual_vs_predicted,
)

# ── Sidebar ──────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="header-gradient">💎 CLV Forecaster</p>', unsafe_allow_html=True)
    st.caption("Customer Lifetime Value Prediction Engine")
    st.markdown("---")

    model_choice = st.selectbox(
        "🤖 Model", ["Ensemble (All 3)", "Random Forest (Primary)", "XGBoost", "Ridge Regression"],
        help="Select which model to use for predictions."
    )

    st.markdown("---")
    st.markdown("### 📊 Dataset Info")
    st.metric("Total Customers", f"{len(df):,}")
    st.metric("Features", f"{len(FEATURES)}")
    st.metric("CLV Range", f"${df['clv_12month'].min():.0f} — ${df['clv_12month'].max():,.0f}")

    st.markdown("---")
    st.markdown(
        "<div style='text-align:center; color:#8892b0; font-size:0.75rem;'>"
        "Built with Streamlit • RF + XGB + Ridge<br>SHAP Explainability</div>",
        unsafe_allow_html=True
    )

# ── Re-predict with selected model ──────────────────────
df_pred = predict_clv(df.copy(), model_choice)

# ── Header ───────────────────────────────────────────────
st.markdown('<p class="header-gradient">💎 Customer Lifetime Value Forecaster</p>', unsafe_allow_html=True)
st.caption("Random Forest + Ridge + XGBoost Ensemble  •  SHAP Explainability  •  What-If Simulation")

# ── Tabs ─────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Portfolio Overview", "🔍 Customer Deep-Dive", "🧪 What-If Simulator",
    "📈 Model Performance", "🗺️ Segmentation", "📋 Customer Table"
])

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 1: Portfolio Overview
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab1:
    stats = get_summary_stats(df_pred)

    # KPI row
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value" style="color:#667eea">${stats['total_predicted_clv']:,.0f}</div>
            <div class="metric-label">Total Portfolio CLV</div></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value" style="color:#2ECC71">${stats['avg_clv']:,.0f}</div>
            <div class="metric-label">Avg CLV per Customer</div></div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value" style="color:#F39C12">${stats['median_clv']:,.0f}</div>
            <div class="metric-label">Median CLV</div></div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value" style="color:#E74C3C">{stats['top_10pct_share']:.1f}%</div>
            <div class="metric-label">Top 10% Revenue Share</div></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Tier breakdown
    st.subheader("🏆 CLV Tier Breakdown")
    tier_cols = st.columns(4)
    for i, (tier_name, info) in enumerate(CLV_TIERS.items()):
        bd = stats['tier_breakdown'].get(tier_name, {'count': 0, 'pct': 0, 'total_clv': 0})
        with tier_cols[i]:
            st.markdown(f"""<div class="metric-card">
                <div style="font-size:1.5rem">{info['emoji']}</div>
                <div class="metric-value" style="color:{info['color']}">{bd['count']:,}</div>
                <div class="metric-label">{info['label']} ({bd['pct']:.1f}%)</div>
                <div style="color:#8892b0; font-size:0.8rem; margin-top:4px">
                    ${bd['total_clv']:,.0f} total CLV</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Charts row
    col_a, col_b = st.columns(2)
    with col_a:
        st.plotly_chart(get_clv_distribution(df_pred), use_container_width=True, key="tab1_distribution")
    with col_b:
        st.plotly_chart(get_tier_revenue_chart(df_pred), use_container_width=True, key="tab1_tier_revenue")

    col_c, col_d = st.columns(2)
    with col_c:
        st.plotly_chart(get_clv_scatter(df_pred), use_container_width=True, key="tab1_scatter")
    with col_d:
        st.plotly_chart(get_cohort_clv_chart(df_pred), use_container_width=True, key="tab1_cohort")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 2: Customer Deep-Dive
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab2:
    st.subheader("🔍 Individual Customer Analysis")

    # Customer selector
    cust_ids = df_pred['customer_id'].tolist()
    selected_id = st.selectbox("Select Customer", cust_ids[:100],
                                help="Choose a customer to analyze")

    cust = df_pred[df_pred['customer_id'] == selected_id].iloc[0]
    clv_val = cust['clv_predicted']
    tier = cust['clv_tier']
    tier_info = CLV_TIERS.get(tier, CLV_TIERS['Low'])

    # Customer header
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        st.markdown(f"""<div class="metric-card">
            <div style="font-size:1.2rem; font-weight:600;">{tier_info['emoji']} {selected_id}</div>
            <div class="metric-value" style="color:{tier_info['color']}">${clv_val:,.2f}</div>
            <div class="metric-label">Predicted 12-Month CLV — {tier_info['label']}</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.metric("Actual CLV", f"${cust['clv_12month']:,.2f}")
        st.metric("RFM Score", f"{cust.get('rfm_score', 'N/A')}/15")
    with c3:
        st.metric("Tenure", f"{cust['tenure_months']} months")
        st.metric("Total Spend", f"${cust['total_spend']:,.2f}")

    # Strategy recommendation
    st.markdown(f"""<div class="strategy-box">
        <strong>💡 Recommended Strategy:</strong> {tier_info['strategy']}
    </div>""", unsafe_allow_html=True)

    # SHAP explanation
    st.markdown("### 🧠 Why This Prediction?")
    scaler = models['scaler']
    cust_features = cust[FEATURES].values.reshape(1, -1)
    cust_scaled = scaler.transform(cust_features)
    shap_data = get_shap_values(cust_scaled[0])

    col_shap, col_detail = st.columns([3, 2])
    with col_shap:
        st.plotly_chart(get_shap_bar_chart(shap_data), use_container_width=True, key="tab2_shap")
    with col_detail:
        st.markdown("**Top Feature Contributions:**")
        for item in shap_data[:7]:
            direction = "🔺" if item['direction'] == 'increases' else "🔻"
            st.markdown(
                f"{direction} **{item['display_name']}** — "
                f"{item['contribution_pct']:.1f}% impact ({item['direction']} CLV)"
            )

    # Customer profile
    st.markdown("### 📋 Full Profile")
    profile_cols = st.columns(4)
    profile_items = [
        ('age', 'Age'), ('income_bracket', 'Income Bracket'),
        ('total_orders', 'Total Orders'), ('recency_days', 'Days Since Purchase'),
        ('nps_score', 'NPS Score'), ('online_ratio', 'Online Ratio'),
        ('return_rate', 'Return Rate'), ('support_tickets', 'Support Tickets'),
    ]
    for i, (feat, label) in enumerate(profile_items):
        with profile_cols[i % 4]:
            val = cust.get(feat, 'N/A')
            if isinstance(val, float):
                if feat in ('online_ratio', 'return_rate'):
                    st.metric(label, f"{val:.1%}")
                else:
                    st.metric(label, f"{val:.1f}")
            else:
                st.metric(label, str(val))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 3: What-If Simulator
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab3:
    st.subheader("🧪 What-If CLV Simulator")
    st.caption("Adjust customer attributes to see how CLV changes in real-time.")

    sim_id = st.selectbox("Select Customer for Simulation", cust_ids[:100], key="sim_cust")
    sim_cust = df_pred[df_pred['customer_id'] == sim_id].iloc[0]

    col_sim1, col_sim2, col_sim3 = st.columns(3)
    with col_sim1:
        new_orders = st.slider("Total Orders", 1, 200, int(sim_cust['total_orders']), key="s_orders")
        new_spend = st.slider("Total Spend ($)", 10, 15000, int(sim_cust['total_spend']), step=50, key="s_spend")
    with col_sim2:
        new_nps = st.slider("NPS Score", 1, 10, int(sim_cust['nps_score']), key="s_nps")
        new_recency = st.slider("Recency (days)", 1, 365, int(sim_cust['recency_days']), key="s_recency")
    with col_sim3:
        new_return = st.slider("Return Rate", 0.0, 0.5, float(sim_cust['return_rate']), 0.01, key="s_return")
        new_tickets = st.slider("Support Tickets", 0, 20, int(sim_cust['support_tickets']), key="s_tickets")

    changes = {
        'total_orders': new_orders, 'total_spend': new_spend,
        'nps_score': new_nps, 'recency_days': new_recency,
        'return_rate': new_return, 'support_tickets': new_tickets,
    }

    result = simulate_clv_change(sim_cust, changes)

    if result:
        rc1, rc2, rc3 = st.columns(3)
        with rc1:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-label">Original CLV</div>
                <div class="metric-value" style="color:#8892b0">${result['original_clv']:,.2f}</div>
                <div class="metric-label">{result['original_tier']}</div>
            </div>""", unsafe_allow_html=True)
        with rc2:
            color = "#2ECC71" if result['change_dollars'] >= 0 else "#E74C3C"
            arrow = "↑" if result['change_dollars'] >= 0 else "↓"
            st.markdown(f"""<div class="metric-card">
                <div class="metric-label">CLV Change</div>
                <div class="metric-value" style="color:{color}">
                    {arrow} ${abs(result['change_dollars']):,.2f}</div>
                <div class="metric-label">{result['change_pct']:+.1f}%</div>
            </div>""", unsafe_allow_html=True)
        with rc3:
            new_tier_color = CLV_TIERS.get(result['new_tier'], CLV_TIERS['Low'])['color']
            st.markdown(f"""<div class="metric-card">
                <div class="metric-label">New CLV</div>
                <div class="metric-value" style="color:{new_tier_color}">${result['new_clv']:,.2f}</div>
                <div class="metric-label">{result['new_tier']}</div>
            </div>""", unsafe_allow_html=True)

        if result['tier_change']:
            st.success(f"🎉 Tier upgrade: **{result['original_tier']}** → **{result['new_tier']}**!")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 4: Model Performance
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab4:
    st.subheader("📈 Model Performance Comparison")
    metrics = models['metrics']

    # Metrics cards
    met_cols = st.columns(4)
    model_names = ['Ridge', 'Random Forest', 'XGBoost', 'Ensemble']
    model_colors = ['#3498DB', '#2ECC71', '#F39C12', '#9B59B6']
    for i, (name, color) in enumerate(zip(model_names, model_colors)):
        m = metrics[name]
        with met_cols[i]:
            st.markdown(f"""<div class="metric-card">
                <div style="font-size:1rem; font-weight:600; color:{color}">{name}</div>
                <div class="metric-value" style="color:{color}">{m['R2']:.4f}</div>
                <div class="metric-label">R² Score</div>
                <div style="color:#8892b0; font-size:0.75rem; margin-top:6px">
                    MAE: ${m['MAE']:,.0f} | RMSE: ${m['RMSE']:,.0f}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.plotly_chart(get_model_comparison_chart(metrics), use_container_width=True, key="tab4_model_comparison")
    with col_m2:
        perf_model = st.selectbox("Actual vs Predicted for:", ['rf', 'xgb', 'ridge', 'ensemble'], key="perf_model")
        st.plotly_chart(get_actual_vs_predicted(df, perf_model), use_container_width=True, key="tab4_actual_vs_pred")

    # Feature importance
    st.markdown("### 🌟 Feature Importance")
    fi_cols = st.columns(2)
    with fi_cols[0]:
        st.plotly_chart(get_feature_importance_chart('rf'), use_container_width=True, key="tab4_fi_rf")
    with fi_cols[1]:
        st.plotly_chart(get_feature_importance_chart('xgb'), use_container_width=True, key="tab4_fi_xgb")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 5: Segmentation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab5:
    st.subheader("🗺️ Customer Segmentation & RFM Analysis")

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.plotly_chart(get_rfm_clv_heatmap(df_pred), use_container_width=True, key="tab5_rfm_heatmap")
    with col_s2:
        st.plotly_chart(get_clv_scatter(df_pred), use_container_width=True, key="tab5_scatter")

    # Tier strategies
    st.markdown("### 💡 Tier-Specific Strategies")
    for tier_name, info in CLV_TIERS.items():
        bd = stats['tier_breakdown'].get(tier_name, {'count': 0, 'total_clv': 0})
        st.markdown(f"""<div class="strategy-box">
            <strong>{info['emoji']} {info['label']} ({tier_name})</strong> — {bd.get('count', 0):,} customers
            <br>{info['strategy']}
        </div>""", unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 6: Customer Table
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab6:
    st.subheader("📋 Full Customer Table")

    # Filters
    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        tier_filter = st.multiselect("Filter by Tier", ['Very High', 'High', 'Medium', 'Low'],
                                      default=['Very High', 'High', 'Medium', 'Low'])
    with fc2:
        clv_range = st.slider("CLV Range ($)", 0, int(df_pred['clv_predicted'].max()),
                               (0, int(df_pred['clv_predicted'].max())), key="clv_range")
    with fc3:
        sort_col = st.selectbox("Sort By", ['clv_predicted', 'clv_12month', 'total_spend',
                                             'rfm_score', 'tenure_months'])

    filtered = df_pred[
        (df_pred['clv_tier'].isin(tier_filter)) &
        (df_pred['clv_predicted'] >= clv_range[0]) &
        (df_pred['clv_predicted'] <= clv_range[1])
    ].sort_values(sort_col, ascending=False)

    st.info(f"Showing **{len(filtered):,}** of {len(df_pred):,} customers")

    display_cols = ['customer_id', 'clv_predicted', 'clv_12month', 'clv_tier',
                    'total_spend', 'total_orders', 'tenure_months', 'rfm_score',
                    'nps_score', 'recency_days', 'region']
    display_cols = [c for c in display_cols if c in filtered.columns]

    # Format dollar columns for display (avoids pandas .style jinja2 dependency)
    display_df = filtered[display_cols].head(500).copy()
    for col in ['clv_predicted', 'clv_12month', 'total_spend']:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: f'${x:,.2f}')

    st.dataframe(display_df, use_container_width=True, height=500)

    # Download
    csv = filtered[display_cols].to_csv(index=False)
    st.download_button("📥 Download Filtered Data (CSV)", csv,
                       "clv_customers_filtered.csv", "text/csv")
