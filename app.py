"""
app.py — CLV Forecaster Streamlit Dashboard
=============================================
Professional 6-tab dashboard for CLV analysis.
Now includes all ML inference and charting logic internally.

Run:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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


# ── Constants & Forecaster Logic ─────────────────────────
FEATURES = [
    'tenure_months', 'total_orders', 'total_spend', 'recency_days',
    'age', 'income_bracket', 'nps_score', 'online_ratio', 'return_rate',
    'support_tickets', 'discount_usage', 'avg_order_value', 'tenure_years',
    'purchase_frequency', 'recency_score', 'frequency_score',
    'monetary_score', 'rfm_score', 'retention_rate', 'churn_rate',
]
TARGET = 'clv_12month'
GROSS_MARGIN = 0.35

CLV_TIERS = {
    'Very High': {'min': 2500, 'color': '#F39C12', 'emoji': '💎',
                  'label': 'Champions',
                  'strategy': 'VIP loyalty program, early product access, dedicated account manager, premium perks'},
    'High':      {'min': 800, 'color': '#2ECC71', 'emoji': '🚀',
                  'label': 'Growers',
                  'strategy': 'Upsell to premium tier, personalised recommendations, milestone rewards'},
    'Medium':    {'min': 200, 'color': '#3498DB', 'emoji': '📈',
                  'label': 'Developing',
                  'strategy': 'Engagement campaigns, product discovery emails, mid-level loyalty points'},
    'Low':       {'min': 0, 'color': '#E74C3C', 'emoji': '⚠️',
                  'label': 'At-Risk / Hibernating',
                  'strategy': 'Win-back offer, re-engagement discount, survey to identify friction point'},
}

FEATURE_META = {
    'tenure_months':      ('Customer Tenure (months)',   '{:.0f} months'),
    'total_orders':       ('Total Orders (historical)',  '{:.0f} orders'),
    'total_spend':        ('Total Historical Spend',     '${:,.2f}'),
    'recency_days':       ('Days Since Last Purchase',   '{:.0f} days'),
    'age':                ('Customer Age',               '{:.0f} years'),
    'income_bracket':     ('Income Bracket (1-4)',       '{:.0f}'),
    'nps_score':          ('NPS Score (1-10)',           '{:.1f}'),
    'online_ratio':       ('Online Purchase Ratio',      '{:.1%}'),
    'return_rate':        ('Return Rate',                '{:.1%}'),
    'support_tickets':    ('Support Tickets',            '{:.0f}'),
    'discount_usage':     ('Discount Usage Rate',        '{:.1%}'),
    'avg_order_value':    ('Avg Order Value',            '${:.2f}'),
    'tenure_years':       ('Tenure (years)',             '{:.1f} yrs'),
    'purchase_frequency': ('Purchase Frequency/yr',      '{:.1f}/yr'),
    'recency_score':      ('Recency Score (1-5)',        '{:.0f}/5'),
    'frequency_score':    ('Frequency Score (1-5)',      '{:.0f}/5'),
    'monetary_score':     ('Monetary Score (1-5)',       '{:.0f}/5'),
    'rfm_score':          ('Combined RFM Score (3-15)',  '{:.0f}/15'),
    'retention_rate':     ('Est. Retention Rate',        '{:.1%}'),
    'churn_rate':         ('Est. Churn Rate',            '{:.1%}'),
}

def get_clv_tier(clv_value: float) -> dict:
    """Return the matching CLV_TIERS entry for a given dollar value."""
    for tier_name in ['Very High', 'High', 'Medium', 'Low']:
        if clv_value >= CLV_TIERS[tier_name]['min']:
            return {**CLV_TIERS[tier_name], 'tier_name': tier_name}
    return {**CLV_TIERS['Low'], 'tier_name': 'Low'}

def _ensure_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features if missing."""
    d = df.copy()
    if 'avg_order_value' not in d.columns:
        d['avg_order_value'] = (d['total_spend'] / d['total_orders']).clip(lower=1.0)
    if 'tenure_years' not in d.columns:
        d['tenure_years'] = d['tenure_months'] / 12.0
    if 'purchase_frequency' not in d.columns:
        d['purchase_frequency'] = (d['total_orders'] / d['tenure_years'].clip(lower=0.08)).clip(upper=365)
    for col in ['recency_score', 'frequency_score', 'monetary_score']:
        if col not in d.columns:
            d[col] = 3
    if 'rfm_score' not in d.columns:
        d['rfm_score'] = d['recency_score'] + d['frequency_score'] + d['monetary_score']
    if 'retention_rate' not in d.columns:
        d['retention_rate'] = ((d['recency_score'] + d['frequency_score']) / 10.0).clip(0.05, 0.95)
    if 'churn_rate' not in d.columns:
        d['churn_rate'] = 1.0 - d['retention_rate']
    return d

def predict_clv(df_new: pd.DataFrame, model_choice: str = 'ensemble') -> pd.DataFrame:
    if 'rf' not in models:
        raise FileNotFoundError("Models not loaded properly.")
    d = _ensure_derived(df_new)
    for f in FEATURES:
        if f not in d.columns:
            d[f] = 0
    X = d[FEATURES].values
    X_scaled = models['scaler'].transform(X)

    preds = {}
    preds['rf'] = np.expm1(models['rf'].predict(X_scaled))
    preds['ridge'] = np.expm1(models['ridge'].predict(X_scaled))
    preds['xgb'] = np.expm1(models['xgb'].predict(X_scaled))
    preds['ensemble'] = np.expm1(
        0.5 * models['rf'].predict(X_scaled) +
        0.3 * models['xgb'].predict(X_scaled) +
        0.2 * models['ridge'].predict(X_scaled))

    choice_key = {'Random Forest (Primary)': 'rf', 'XGBoost': 'xgb',
                  'Ridge Regression': 'ridge', 'Ensemble (All 3)': 'ensemble'}
    key = choice_key.get(model_choice, model_choice)
    d['clv_predicted'] = preds.get(key, preds['ensemble'])
    d['model_used'] = key
    tiers = [get_clv_tier(v) for v in d['clv_predicted']]
    d['clv_tier'] = [t['tier_name'] for t in tiers]
    d['tier_color'] = [t['color'] for t in tiers]
    d['tier_emoji'] = [t['emoji'] for t in tiers]
    d['tier_label'] = [t['label'] for t in tiers]
    d['tier_strategy'] = [t['strategy'] for t in tiers]
    return d.sort_values('clv_predicted', ascending=False).reset_index(drop=True)

def get_shap_values(customer_row_scaled: np.ndarray, n_features: int = 10) -> list:
    if 'shap' not in models:
        return []
    raw = models['shap'].shap_values(customer_row_scaled.reshape(1, -1))
    sv = raw[0] if isinstance(raw, list) else raw[0]
    total = np.abs(sv).sum()
    pairs = sorted(zip(FEATURES, sv), key=lambda x: abs(x[1]), reverse=True)
    result = []
    for feat, val in pairs[:n_features]:
        meta = FEATURE_META.get(feat, (feat, '{:.2f}'))
        result.append({
            'feature': feat,
            'display_name': meta[0],
            'shap_value': float(val),
            'direction': 'increases' if val > 0 else 'decreases',
            'contribution_pct': float(abs(val) / total * 100) if total > 0 else 0,
        })
    return result

def simulate_clv_change(customer_row: pd.Series, changes: dict) -> dict:
    if 'rf' not in models:
        return {}
    original = customer_row.copy()
    modified = customer_row.copy()
    for k, v in changes.items():
        if k in modified.index:
            modified[k] = v

    modified['avg_order_value'] = max(modified['total_spend'] / max(modified['total_orders'], 1), 1.0)
    modified['tenure_years'] = modified['tenure_months'] / 12.0
    modified['purchase_frequency'] = min(modified['total_orders'] / max(modified['tenure_years'], 0.08), 365)
    modified['retention_rate'] = np.clip(
        (modified.get('recency_score', 3) + modified.get('frequency_score', 3)) / 10.0, 0.05, 0.95)
    modified['churn_rate'] = 1.0 - modified['retention_rate']

    orig_df = pd.DataFrame([original[FEATURES]])
    mod_df = pd.DataFrame([modified[FEATURES]])
    orig_scaled = models['scaler'].transform(orig_df)
    mod_scaled = models['scaler'].transform(mod_df)

    orig_clv = float(np.expm1(models['rf'].predict(orig_scaled)[0]))
    new_preds = {
        'rf': float(np.expm1(models['rf'].predict(mod_scaled)[0])),
        'ridge': float(np.expm1(models['ridge'].predict(mod_scaled)[0])),
        'xgb': float(np.expm1(models['xgb'].predict(mod_scaled)[0])),
    }
    new_clv = new_preds['rf']
    orig_tier = get_clv_tier(orig_clv)
    new_tier = get_clv_tier(new_clv)

    return {
        'original_clv': orig_clv, 'new_clv': new_clv,
        'change_dollars': new_clv - orig_clv,
        'change_pct': ((new_clv - orig_clv) / orig_clv * 100) if orig_clv > 0 else 0,
        'original_tier': orig_tier['tier_name'], 'new_tier': new_tier['tier_name'],
        'tier_change': orig_tier['tier_name'] != new_tier['tier_name'],
        'model_predictions': new_preds,
    }

def get_summary_stats(df: pd.DataFrame) -> dict:
    col = 'clv_predicted' if 'clv_predicted' in df.columns else 'clv_predicted_rf'
    total_clv = df[col].sum()
    top10 = df.nlargest(int(len(df) * 0.1), col)[col].sum()

    tier_col = 'clv_tier' if 'clv_tier' in df.columns else 'clv_segment'
    tier_bd = {}
    if tier_col in df.columns:
        for t in ['Very High', 'High', 'Medium', 'Low']:
            mask = df[tier_col] == t
            tier_bd[t] = {'count': int(mask.sum()),
                          'pct': float(mask.sum() / len(df) * 100),
                          'total_clv': float(df.loc[mask, col].sum())}

    best_region = ''
    if 'region' in df.columns:
        best_region = df.groupby('region')[col].mean().idxmax()
    best_income = ''
    if 'income_bracket' in df.columns:
        best_income = str(df.groupby('income_bracket')[col].mean().idxmax())

    return {
        'total_customers': len(df), 'total_predicted_clv': float(total_clv),
        'avg_clv': float(df[col].mean()), 'median_clv': float(df[col].median()),
        'top_10pct_clv': float(top10),
        'top_10pct_share': float(top10 / total_clv * 100) if total_clv > 0 else 0,
        'tier_breakdown': tier_bd, 'best_region': best_region,
        'best_income_bracket': best_income,
    }

def get_clv_scatter(df: pd.DataFrame) -> go.Figure:
    col = 'clv_predicted' if 'clv_predicted' in df.columns else 'clv_predicted_rf'
    tier_col = 'clv_tier' if 'clv_tier' in df.columns else 'clv_segment'
    freq = df['purchase_frequency'] if 'purchase_frequency' in df.columns else pd.Series([10]*len(df))
    norm_size = 4 + 12 * (freq - freq.min()) / (freq.max() - freq.min() + 1e-9)

    fig = px.scatter(df, x='total_spend', y=col, color=tier_col,
                     size=norm_size, hover_data=['customer_id', 'tenure_months', 'rfm_score', 'total_orders'],
                     color_discrete_map={'Very High': '#F39C12', 'High': '#2ECC71',
                                         'Medium': '#3498DB', 'Low': '#E74C3C'},
                     template='plotly_dark',
                     title='Customer Spend vs Predicted CLV')
    x_range = [0, df['total_spend'].max() * 1.05]
    fig.add_trace(go.Scatter(x=x_range, y=[v * GROSS_MARGIN for v in x_range],
                             mode='lines', line=dict(dash='dash', color='gray'),
                             name=f'y = x × {GROSS_MARGIN}', showlegend=True))
    fig.update_layout(height=500, xaxis_title='Total Spend ($)', yaxis_title='Predicted CLV ($)')
    return fig

def get_rfm_clv_heatmap(df: pd.DataFrame) -> go.Figure:
    col = 'clv_predicted' if 'clv_predicted' in df.columns else 'clv_predicted_rf'
    pivot = df.pivot_table(values=col, index='recency_score', columns='frequency_score', aggfunc='mean')
    fig = px.imshow(pivot, color_continuous_scale='Viridis', template='plotly_dark',
                    title='Mean CLV by Recency × Frequency Score',
                    labels=dict(x='Frequency Score', y='Recency Score', color='Mean CLV ($)'))
    fig.update_layout(height=450)
    return fig

def get_model_comparison_chart(metrics: dict) -> go.Figure:
    fig = make_subplots(rows=2, cols=2, subplot_titles=['MAE ($)', 'RMSE ($)', 'R²', 'MAPE (%)'])
    colors = {'Ridge': '#3498DB', 'Random Forest': '#2ECC71', 'XGBoost': '#F39C12', 'Ensemble': '#9B59B6'}
    names = list(metrics.keys())
    for i, (metric, r, c) in enumerate([('MAE', 1, 1), ('RMSE', 1, 2), ('R2', 2, 1), ('MAPE', 2, 2)]):
        vals = [metrics[n][metric] for n in names]
        fig.add_trace(go.Bar(x=names, y=vals, marker_color=[colors.get(n, '#888') for n in names],
                             showlegend=False), row=r, col=c)
    fig.update_layout(template='plotly_dark', height=500, title_text='Model Performance Comparison')
    return fig

def get_shap_bar_chart(shap_data: list) -> go.Figure:
    if not shap_data:
        return go.Figure().update_layout(template='plotly_dark', title='No SHAP data available')
    features = [d['display_name'] for d in reversed(shap_data)]
    values = [d['shap_value'] for d in reversed(shap_data)]
    colors = ['#E74C3C' if v > 0 else '#3498DB' for v in values]
    fig = go.Figure(go.Bar(y=features, x=values, orientation='h', marker_color=colors))
    fig.add_vline(x=0, line_color='white', line_width=1)
    fig.update_layout(template='plotly_dark', height=400,
                      title="Feature Contributions — This Customer's CLV",
                      xaxis_title='SHAP Value (log CLV impact)',
                      yaxis_title='')
    return fig

def get_clv_distribution(df: pd.DataFrame) -> go.Figure:
    col = 'clv_predicted' if 'clv_predicted' in df.columns else 'clv_predicted_rf'
    tier_col = 'clv_tier' if 'clv_tier' in df.columns else 'clv_segment'
    fig = px.histogram(df, x=col, color=tier_col, nbins=60, log_x=True,
                       color_discrete_map={'Very High': '#F39C12', 'High': '#2ECC71',
                                           'Medium': '#3498DB', 'Low': '#E74C3C'},
                       template='plotly_dark',
                       title='Predicted CLV Distribution — All Customers')
    for thresh in [200, 800, 2500]:
        fig.add_vline(x=thresh, line_dash='dash', line_color='rgba(255,255,255,0.4)',
                      annotation_text=f'${thresh:,}')
    fig.update_layout(height=400, xaxis_title='Predicted CLV ($, log scale)', yaxis_title='Count')
    return fig

def get_feature_importance_chart(model_name: str = 'rf') -> go.Figure:
    if model_name == 'rf' and 'rf' in models:
        imp = models['rf'].feature_importances_
        title = 'Feature Importance — Random Forest'
    elif model_name == 'xgb' and 'xgb' in models:
        imp = models['xgb'].feature_importances_
        title = 'Feature Importance — XGBoost'
    elif model_name == 'ridge' and 'ridge' in models:
        imp = np.abs(models['ridge'].coef_)
        imp = imp / imp.sum()
        title = 'Feature Importance — Ridge (|coefficients|)'
    else:
        return go.Figure().update_layout(template='plotly_dark', title='Model not available')

    imp_df = pd.DataFrame({'feature': FEATURES, 'importance': imp}).sort_values('importance', ascending=True).tail(15)
    display_names = [FEATURE_META.get(f, (f,))[0] for f in imp_df['feature']]
    fig = go.Figure(go.Bar(y=display_names, x=imp_df['importance'].values, orientation='h',
                           marker_color='#2ECC71' if model_name == 'rf' else '#F39C12' if model_name == 'xgb' else '#3498DB'))
    fig.update_layout(template='plotly_dark', height=450, title=title,
                      xaxis_title='Importance', yaxis_title='')
    return fig

def get_tier_revenue_chart(df: pd.DataFrame) -> go.Figure:
    col = 'clv_predicted' if 'clv_predicted' in df.columns else 'clv_predicted_rf'
    tier_col = 'clv_tier' if 'clv_tier' in df.columns else 'clv_segment'
    tier_order = ['Very High', 'High', 'Medium', 'Low']
    grp = df.groupby(tier_col).agg(total_clv=(col, 'sum'), count=(col, 'count')).reindex(tier_order).fillna(0)
    total = grp['total_clv'].sum()
    colors = ['#F39C12', '#2ECC71', '#3498DB', '#E74C3C']

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=grp.index, y=grp['total_clv'], name='Total CLV ($)',
                         marker_color=colors, text=[f'{v/total*100:.1f}%' if total > 0 else '0%' for v in grp['total_clv']],
                         textposition='auto'), secondary_y=False)
    fig.add_trace(go.Scatter(x=grp.index, y=grp['count'], name='Customer Count',
                             mode='lines+markers', line=dict(color='white', width=2),
                             marker=dict(size=10)), secondary_y=True)
    fig.update_layout(template='plotly_dark', height=400,
                      title='Revenue Contribution and Customer Count by CLV Tier')
    fig.update_yaxes(title_text='Total Predicted CLV ($)', secondary_y=False)
    fig.update_yaxes(title_text='Customer Count', secondary_y=True)
    return fig

def get_cohort_clv_chart(df: pd.DataFrame) -> go.Figure:
    col = 'clv_predicted' if 'clv_predicted' in df.columns else 'clv_predicted_rf'
    bins = [0, 6, 12, 24, 48, 999]
    labels = ['0-6 mo', '6-12 mo', '12-24 mo', '24-48 mo', '48+ mo']
    df = df.copy()
    df['tenure_cohort'] = pd.cut(df['tenure_months'], bins=bins, labels=labels)
    fig = px.box(df, x='tenure_cohort', y=col, color='tenure_cohort', log_y=True,
                 category_orders={'tenure_cohort': labels}, template='plotly_dark',
                 title='CLV Distribution by Customer Tenure Cohort',
                 color_discrete_sequence=['#E74C3C', '#E67E22', '#F1C40F', '#2ECC71', '#3498DB'])
    fig.update_layout(height=400, xaxis_title='Tenure Cohort', yaxis_title='Predicted CLV ($ log scale)',
                      showlegend=False)
    return fig

def get_actual_vs_predicted(df: pd.DataFrame, model_choice: str = 'rf') -> go.Figure:
    pred_col_map = {'rf': 'clv_predicted_rf', 'xgb': 'clv_predicted_xgb',
                    'ridge': 'clv_predicted_ridge', 'ensemble': 'clv_predicted_ensemble'}
    pred_col = pred_col_map.get(model_choice, 'clv_predicted_rf')
    if pred_col not in df.columns:
        pred_col = 'clv_predicted' if 'clv_predicted' in df.columns else 'clv_predicted_rf'
    tier_col = 'clv_tier' if 'clv_tier' in df.columns else 'clv_segment'

    from sklearn.metrics import r2_score as _r2
    r2 = _r2(df['clv_12month'], df[pred_col]) if len(df) > 1 else 0

    fig = px.scatter(df, x='clv_12month', y=pred_col, color=tier_col, log_x=True, log_y=True,
                     color_discrete_map={'Very High': '#F39C12', 'High': '#2ECC71',
                                         'Medium': '#3498DB', 'Low': '#E74C3C'},
                     template='plotly_dark',
                     title=f'Actual vs Predicted CLV — {model_choice.upper()} (R²={r2:.3f})')
    rng = [df['clv_12month'].min(), df['clv_12month'].max()]
    fig.add_trace(go.Scatter(x=rng, y=rng, mode='lines', line=dict(dash='dash', color='white'),
                             name='Perfect (y=x)', showlegend=True))
    fig.add_trace(go.Scatter(x=rng, y=[v * 1.25 for v in rng], mode='lines',
                             line=dict(dash='dot', color='rgba(255,255,255,0.2)'),
                             name='+25%', showlegend=False))
    fig.add_trace(go.Scatter(x=rng, y=[v * 0.75 for v in rng], mode='lines',
                             line=dict(dash='dot', color='rgba(255,255,255,0.2)'),
                             name='-25%', showlegend=False))
    fig.update_layout(height=500, xaxis_title='Actual CLV ($)', yaxis_title='Predicted CLV ($)')
    return fig


# ── Load everything ──────────────────────────────────────
@st.cache_resource
def load_models():
    """Load all trained models and artifacts."""
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
    return pd.read_csv("data/customers_clv.csv")

try:
    models = load_models()
    df = load_data()
    DATA_READY = True
except Exception as e:
    DATA_READY = False
    st.error(f"Failed to load data/models: {e}")
    st.info("Please run the `Full_ML_Code.ipynb` notebook to generate models and data before starting the app.")
    st.stop()

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
    cust_ids = df_pred['customer_id'].tolist()
    selected_id = st.selectbox("Select Customer", cust_ids[:100], help="Choose a customer to analyze")

    cust = df_pred[df_pred['customer_id'] == selected_id].iloc[0]
    clv_val = cust['clv_predicted']
    tier = cust['clv_tier']
    tier_info = CLV_TIERS.get(tier, CLV_TIERS['Low'])

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

    st.markdown(f"""<div class="strategy-box">
        <strong>💡 Recommended Strategy:</strong> {tier_info['strategy']}
    </div>""", unsafe_allow_html=True)

    st.markdown("### 🧠 Why This Prediction?")
    cust_features = cust[FEATURES].values.reshape(1, -1)
    cust_scaled = models['scaler'].transform(cust_features)
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

    display_df = filtered[display_cols].head(500).copy()
    for col in ['clv_predicted', 'clv_12month', 'total_spend']:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: f'${x:,.2f}')

    st.dataframe(display_df, use_container_width=True, height=500)

    csv = filtered[display_cols].to_csv(index=False)
    st.download_button("📥 Download Filtered Data (CSV)", csv,
                       "clv_customers_filtered.csv", "text/csv")
