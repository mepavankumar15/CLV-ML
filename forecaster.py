"""
forecaster.py — CLV Inference, SHAP Explanation & Chart Functions
=================================================================
ML brain for the CLV Forecaster. NO Streamlit imports.
Provides prediction, explanation, simulation, and visualization.
"""

import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Constants ────────────────────────────────────────────
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

# ── Model Loading ────────────────────────────────────────
try:
    rf_model = joblib.load("models/rf_model.pkl")
    ridge_model = joblib.load("models/ridge_model.pkl")
    xgb_model = joblib.load("models/xgb_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    shap_explainer = joblib.load("models/shap_explainer.pkl")
    MODELS_LOADED = True
except FileNotFoundError:
    rf_model = ridge_model = xgb_model = scaler = shap_explainer = None
    MODELS_LOADED = False


def get_clv_tier(clv_value: float) -> dict:
    """Return the matching CLV_TIERS entry for a given dollar value."""
    for tier_name in ['Very High', 'High', 'Medium', 'Low']:
        if clv_value >= CLV_TIERS[tier_name]['min']:
            return {**CLV_TIERS[tier_name], 'tier_name': tier_name}
    return {**CLV_TIERS['Low'], 'tier_name': 'Low'}


def load_scored_data() -> pd.DataFrame:
    """Load the scored customer dataset with predictions."""
    try:
        df = pd.read_csv("data/customers_clv.csv")
    except FileNotFoundError:
        raise FileNotFoundError("Run train_model.py first to generate scored data.")
    if 'clv_tier' not in df.columns:
        df['clv_tier'] = pd.cut(
            df['clv_predicted_rf'], bins=[0, 200, 800, 2500, 1e9],
            labels=['Low', 'Medium', 'High', 'Very High'])
    return df


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
    """
    Predict CLV for new customers.

    Parameters
    ----------
    df_new : pd.DataFrame  — raw customer data
    model_choice : str      — 'rf', 'xgb', 'ridge', or 'ensemble'

    Returns
    -------
    pd.DataFrame sorted by clv_predicted descending
    """
    if not MODELS_LOADED:
        raise FileNotFoundError("Run train_model.py first to generate models.")
    d = _ensure_derived(df_new)
    for f in FEATURES:
        if f not in d.columns:
            d[f] = 0
    X = d[FEATURES].values
    X_scaled = scaler.transform(X)

    preds = {}
    preds['rf'] = np.expm1(rf_model.predict(X_scaled))
    preds['ridge'] = np.expm1(ridge_model.predict(X_scaled))
    preds['xgb'] = np.expm1(xgb_model.predict(X_scaled))
    preds['ensemble'] = np.expm1(
        0.5 * rf_model.predict(X_scaled) +
        0.3 * xgb_model.predict(X_scaled) +
        0.2 * ridge_model.predict(X_scaled))

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
    """
    Compute SHAP values for a single customer.

    Returns list of dicts with feature, display_name, shap_value, direction, contribution_pct.
    """
    if not MODELS_LOADED:
        return []
    raw = shap_explainer.shap_values(customer_row_scaled.reshape(1, -1))
    # SHAP <0.41: returns list of arrays; SHAP >=0.41 regressor: returns 2D array
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
    """
    Simulate CLV impact of proposed changes to a customer.

    Parameters
    ----------
    customer_row : pd.Series — original customer data
    changes : dict — e.g. {'total_orders': +5, 'nps_score': 8}

    Returns
    -------
    dict with original_clv, new_clv, change_dollars, change_pct, tier info
    """
    if not MODELS_LOADED:
        return {}
    original = customer_row.copy()
    modified = customer_row.copy()
    for k, v in changes.items():
        if k in modified.index:
            modified[k] = v

    # Recompute derived features
    modified['avg_order_value'] = max(modified['total_spend'] / max(modified['total_orders'], 1), 1.0)
    modified['tenure_years'] = modified['tenure_months'] / 12.0
    modified['purchase_frequency'] = min(modified['total_orders'] / max(modified['tenure_years'], 0.08), 365)
    modified['retention_rate'] = np.clip(
        (modified.get('recency_score', 3) + modified.get('frequency_score', 3)) / 10.0, 0.05, 0.95)
    modified['churn_rate'] = 1.0 - modified['retention_rate']

    orig_df = pd.DataFrame([original[FEATURES]])
    mod_df = pd.DataFrame([modified[FEATURES]])
    orig_scaled = scaler.transform(orig_df)
    mod_scaled = scaler.transform(mod_df)

    orig_clv = float(np.expm1(rf_model.predict(orig_scaled)[0]))
    new_preds = {
        'rf': float(np.expm1(rf_model.predict(mod_scaled)[0])),
        'ridge': float(np.expm1(ridge_model.predict(mod_scaled)[0])),
        'xgb': float(np.expm1(xgb_model.predict(mod_scaled)[0])),
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
    """Compute portfolio-level summary statistics."""
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


# ── Chart Functions ──────────────────────────────────────

def get_clv_scatter(df: pd.DataFrame) -> go.Figure:
    """Scatter: total_spend vs clv_predicted, colored by tier."""
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
    """Heatmap: mean CLV by Recency × Frequency score."""
    col = 'clv_predicted' if 'clv_predicted' in df.columns else 'clv_predicted_rf'
    pivot = df.pivot_table(values=col, index='recency_score', columns='frequency_score', aggfunc='mean')
    fig = px.imshow(pivot, color_continuous_scale='Viridis', template='plotly_dark',
                    title='Mean CLV by Recency × Frequency Score',
                    labels=dict(x='Frequency Score', y='Recency Score', color='Mean CLV ($)'))
    fig.update_layout(height=450)
    return fig


def get_model_comparison_chart(metrics: dict) -> go.Figure:
    """Grouped bar chart comparing models on MAE, RMSE, R², MAPE."""
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
    """Horizontal bar chart of SHAP values for one customer."""
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
    """Histogram of predicted CLV with tier color segments."""
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
    """Bar chart of feature importances for a given model."""
    if model_name == 'rf' and rf_model is not None:
        imp = rf_model.feature_importances_
        title = 'Feature Importance — Random Forest'
    elif model_name == 'xgb' and xgb_model is not None:
        imp = xgb_model.feature_importances_
        title = 'Feature Importance — XGBoost'
    elif model_name == 'ridge' and ridge_model is not None:
        imp = np.abs(ridge_model.coef_)
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
    """Stacked bar of CLV contribution and customer count by tier."""
    col = 'clv_predicted' if 'clv_predicted' in df.columns else 'clv_predicted_rf'
    tier_col = 'clv_tier' if 'clv_tier' in df.columns else 'clv_segment'
    tier_order = ['Very High', 'High', 'Medium', 'Low']
    grp = df.groupby(tier_col).agg(total_clv=(col, 'sum'), count=(col, 'count')).reindex(tier_order).fillna(0)
    total = grp['total_clv'].sum()
    colors = ['#F39C12', '#2ECC71', '#3498DB', '#E74C3C']

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=grp.index, y=grp['total_clv'], name='Total CLV ($)',
                         marker_color=colors, text=[f'{v/total*100:.1f}%' for v in grp['total_clv']],
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
    """Box plot of CLV by tenure cohort."""
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
    """Scatter: actual vs predicted CLV with perfect prediction line."""
    pred_col_map = {'rf': 'clv_predicted_rf', 'xgb': 'clv_predicted_xgb',
                    'ridge': 'clv_predicted_ridge', 'ensemble': 'clv_predicted_ensemble'}
    pred_col = pred_col_map.get(model_choice, 'clv_predicted_rf')
    if pred_col not in df.columns:
        pred_col = 'clv_predicted' if 'clv_predicted' in df.columns else 'clv_predicted_rf'
    tier_col = 'clv_tier' if 'clv_tier' in df.columns else 'clv_segment'

    from sklearn.metrics import r2_score as _r2
    r2 = _r2(df['clv_12month'], df[pred_col])

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
