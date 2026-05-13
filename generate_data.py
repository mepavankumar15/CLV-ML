"""
generate_data.py — Synthetic Customer CLV Dataset Generator
============================================================
Generates 3,000 synthetic retail customers across 4 CLV tiers:
  Champions (600), Growers (900), At-Risk (800), Hibernating (700)

Each customer has demographics, transactional RFM features,
behavioral features, and a computed 12-month CLV target.

Run:  python generate_data.py
Output: data/customers.csv
"""

import numpy as np
import pandas as pd
import os

# ── Constants ────────────────────────────────────────────
np.random.seed(42)
N = 3000
GROSS_MARGIN = 0.35  # Assumed 35% gross margin for retail


def generate_tier(n, tier_name, config):
    """
    Generate n customers for a single CLV tier.

    Parameters
    ----------
    n : int
        Number of customers to generate.
    tier_name : str
        Name of the tier (e.g. 'Champions').
    config : dict
        Tier-specific parameter ranges.

    Returns
    -------
    pd.DataFrame
        DataFrame with raw customer features for this tier.
    """
    data = {
        'tenure_months':     np.random.randint(config['tenure'][0], config['tenure'][1], n),
        'total_orders':      np.random.randint(config['orders'][0], config['orders'][1], n),
        'total_spend':       np.clip(
            np.random.lognormal(mean=np.log(config['spend_mean']),
                                sigma=config['spend_sigma'], size=n),
            config['spend_clip'][0], config['spend_clip'][1]
        ),
        'recency_days':      np.random.randint(config['recency'][0], config['recency'][1], n),
        'age':               np.random.randint(config['age'][0], config['age'][1], n),
        'income_bracket':    np.random.choice(config['income'], size=n),
        'nps_score':         np.random.randint(config['nps'][0], config['nps'][1], n),
        'online_ratio':      np.random.uniform(config['online'][0], config['online'][1], n),
        'return_rate':       np.random.uniform(config['return_rate'][0], config['return_rate'][1], n),
        'support_tickets':   np.random.randint(config['tickets'][0], config['tickets'][1], n),
        'discount_usage':    np.random.uniform(config['discount'][0], config['discount'][1], n),
        'category_preference': np.random.choice(
            config['categories'], size=n,
            p=config.get('cat_probs', None)
        ),
        'region':            np.random.choice(['North', 'South', 'East', 'West'], size=n),
        'gender':            np.random.choice(['M', 'F'], size=n),
        'clv_tier':          tier_name,
    }
    return pd.DataFrame(data)


# ── Tier Configurations ─────────────────────────────────

tier1_config = {
    'tenure':      (24, 84),
    'orders':      (40, 200),
    'spend_mean':  8000,
    'spend_sigma': 0.4,
    'spend_clip':  (2500, 15000),
    'recency':     (1, 30),
    'age':         (30, 65),
    'income':      [3, 4, 4, 4],
    'nps':         (8, 11),
    'online':      (0.3, 0.8),
    'return_rate': (0.01, 0.08),
    'tickets':     (0, 5),
    'discount':    (0.0, 0.15),
    'categories':  ['Electronics', 'Luxury', 'Home'],
    'cat_probs':   [0.4, 0.4, 0.2],
}

tier2_config = {
    'tenure':      (6, 36),
    'orders':      (12, 60),
    'spend_mean':  1500,
    'spend_sigma': 0.45,
    'spend_clip':  (800, 2500),
    'recency':     (15, 90),
    'age':         (25, 55),
    'income':      [2, 3, 3],
    'nps':         (6, 10),
    'online':      (0.4, 0.9),
    'return_rate': (0.05, 0.15),
    'tickets':     (1, 8),
    'discount':    (0.1, 0.35),
    'categories':  ['Clothing', 'Sports', 'Tech'],
    'cat_probs':   [0.4, 0.3, 0.3],
}

tier3_config = {
    'tenure':      (12, 48),
    'orders':      (4, 20),
    'spend_mean':  450,
    'spend_sigma': 0.5,
    'spend_clip':  (200, 800),
    'recency':     (60, 200),
    'age':         (20, 70),
    'income':      [1, 2, 2],
    'nps':         (4, 8),
    'online':      (0.2, 0.6),
    'return_rate': (0.1, 0.3),
    'tickets':     (3, 15),
    'discount':    (0.3, 0.7),
    'categories':  ['Grocery', 'Pharmacy', 'Clothing'],
    'cat_probs':   None,
}

tier4_config = {
    'tenure':      (1, 24),
    'orders':      (1, 8),
    'spend_mean':  80,
    'spend_sigma': 0.6,
    'spend_clip':  (10, 200),
    'recency':     (180, 365),
    'age':         (18, 75),
    'income':      [1, 1, 2],
    'nps':         (1, 7),
    'online':      (0.1, 0.5),
    'return_rate': (0.2, 0.5),
    'tickets':     (2, 20),
    'discount':    (0.4, 1.0),
    'categories':  ['Grocery', 'Pharmacy'],
    'cat_probs':   None,
}

# ── STEP 1: Generate each tier ──────────────────────────
print("Generating synthetic CLV dataset...")

df_champions   = generate_tier(600,  'Champions',   tier1_config)
df_growers     = generate_tier(900,  'Growers',     tier2_config)
df_atrisk      = generate_tier(800,  'At-Risk',     tier3_config)
df_hibernating = generate_tier(700,  'Hibernating', tier4_config)

# ── STEP 2: Combine all tiers ───────────────────────────
df = pd.concat([df_champions, df_growers, df_atrisk, df_hibernating],
               ignore_index=True)

# Assign zero-padded customer IDs
df['customer_id'] = [f"CUST_{i+1:05d}" for i in range(len(df))]

# Shuffle to mix tiers randomly
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"  Generated {len(df):,} customers across 4 tiers")

# ── STEP 3: Compute derived features ────────────────────
print("Computing derived features...")

# (a) Average order value = total_spend / total_orders
df['avg_order_value'] = (df['total_spend'] / df['total_orders']).clip(lower=1.0)

# (b) Tenure in years
df['tenure_years'] = df['tenure_months'] / 12.0

# (c) Purchase frequency = orders per year
df['purchase_frequency'] = (df['total_orders'] / df['tenure_years']).clip(upper=365.0)

# (d) Recency score: quintile score 1-5 (5 = most recent)
df['recency_score'] = pd.qcut(
    df['recency_days'], q=5, labels=[5, 4, 3, 2, 1], duplicates='drop'
).astype(int)

# (e) Frequency score: quintile score 1-5 (5 = most frequent)
df['frequency_score'] = pd.qcut(
    df['total_orders'], q=5, labels=[1, 2, 3, 4, 5], duplicates='drop'
).astype(int)

# (f) Monetary score: quintile score 1-5 (5 = highest spend)
df['monetary_score'] = pd.qcut(
    df['total_spend'], q=5, labels=[1, 2, 3, 4, 5], duplicates='drop'
).astype(int)

# (g) Combined RFM score (range 3-15)
df['rfm_score'] = df['recency_score'] + df['frequency_score'] + df['monetary_score']

# (h) Retention rate — estimated from recency and frequency scores
base_retention = (df['recency_score'] + df['frequency_score']) / 10.0
noise = np.random.normal(0, 0.05, size=len(df))
df['retention_rate'] = (base_retention + noise).clip(0.05, 0.95)

# (i) Churn rate = 1 - retention_rate
df['churn_rate'] = 1.0 - df['retention_rate']

# (j) CLV 12-month TARGET
#     Formula: (avg_order_value × purchase_frequency × GROSS_MARGIN) / churn_rate
#     Add lognormal noise for realism
raw_clv = (df['avg_order_value'] * df['purchase_frequency'] * GROSS_MARGIN) / df['churn_rate']
clv_noise = np.random.lognormal(0, 0.15, size=len(df))
df['clv_12month'] = (raw_clv * clv_noise).clip(5.0, 50000.0).round(2)

# (k) CLV segment labels
df['clv_segment'] = pd.cut(
    df['clv_12month'],
    bins=[0, 200, 800, 2500, 50001],
    labels=['Low', 'Medium', 'High', 'Very High']
)

print("  Derived features computed [OK]")

# ── STEP 4: Save dataset ────────────────────────────────
os.makedirs("data", exist_ok=True)
df.to_csv("data/customers.csv", index=False)

# ── STEP 5: Print summary ───────────────────────────────
print("\n" + "=" * 55)
print("   CLV Dataset Generated")
print("=" * 55)
print(f"  Total customers:     {N:,}")
print(f"  Columns:             {len(df.columns)}")
print(f"  CLV range:           ${df.clv_12month.min():.2f} — ${df.clv_12month.max():,.2f}")
print(f"  Mean CLV:            ${df.clv_12month.mean():,.2f}")
print(f"  Median CLV:          ${df.clv_12month.median():,.2f}")
print(f"  CLV Segments:")
print(f"    Low (<$200):       {(df.clv_segment == 'Low').sum():,}")
print(f"    Medium ($200-800): {(df.clv_segment == 'Medium').sum():,}")
print(f"    High ($800-2500):  {(df.clv_segment == 'High').sum():,}")
print(f"    Very High (>$2500):{(df.clv_segment == 'Very High').sum():,}")
print(f"\n  Data saved to data/customers.csv [OK]")
print("=" * 55)
