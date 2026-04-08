import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import joblib
import json
import warnings
from pathlib import Path
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title  = "predictive-forecasting-of-care-load-and-placement-demand",
    page_icon   = "🏠",
    layout      = "wide",
    initial_sidebar_state = "expanded"
)

# ── Paths ─────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent.parent
MODELS_DIR  = BASE_DIR / 'models'
DATA_DIR    = BASE_DIR / 'data' / 'processed'
FIGURES_DIR = BASE_DIR / 'reports' / 'figures'


# ── Load models and config ────────────────────────────────────
@st.cache_resource
def load_models():
    careload_model   = joblib.load(MODELS_DIR / 'best_model_recent.joblib')
    discharge_model  = joblib.load(MODELS_DIR / 'best_discharge_model_recent.joblib')
    scaler_care      = joblib.load(MODELS_DIR / 'scaler_recent.joblib')
    scaler_discharge = joblib.load(MODELS_DIR / 'scaler_discharge_recent.joblib')
    return careload_model, discharge_model, scaler_care, scaler_discharge

@st.cache_resource
def load_configs():
    with open(MODELS_DIR / 'feature_columns_recent.json') as f:
        care_features = json.load(f)
    with open(MODELS_DIR / 'discharge_feature_columns_recent.json') as f:
        discharge_features = json.load(f)
    with open(MODELS_DIR / 'thresholds.json') as f:
        thresholds = json.load(f)
    with open(MODELS_DIR / 'model_config.json') as f:
        model_config = json.load(f)
    return care_features, discharge_features, thresholds, model_config

@st.cache_data
def load_data():
    df = pd.read_csv(
        DATA_DIR / 'featured_data.csv',
        index_col='Date', parse_dates=True
    )
    return df

@st.cache_data
def load_results():
    complete = pd.read_csv(MODELS_DIR / 'complete_results_final.csv', index_col=0)
    baseline = pd.read_csv(MODELS_DIR / 'baseline_results.csv',       index_col=0)
    complete.index.name = 'Model'
    baseline.index.name = 'Model'
    return complete, baseline

careload_model, discharge_model, scaler_care, scaler_discharge = load_models()
care_features, discharge_features, thresholds, model_config    = load_configs()
df                                                              = load_data()
complete_results, baseline_results                             = load_results()

STRESS_THRESHOLD   = thresholds['stress_threshold']
CRITICAL_THRESHOLD = thresholds['critical_threshold']
HISTORICAL_MEAN    = thresholds['historical_mean']

# ── Helper functions ──────────────────────────────────────────
def get_alert_level(value: float) -> str:
    if value >= CRITICAL_THRESHOLD:
        return 'CRITICAL'
    elif value >= STRESS_THRESHOLD:
        return 'STRESS'
    else:
        return 'NORMAL'

def get_alert_color(level: str) -> str:
    return {'CRITICAL': '#e74c3c', 'STRESS': '#f39c12', 'NORMAL': '#27ae60'}[level]

def get_alert_emoji(level: str) -> str:
    return {'CRITICAL': '🔴', 'STRESS': '🟡', 'NORMAL': '🟢'}[level]

def build_feature_row(
    lag1, lag2, lag3, lag7, lag14,
    roll7_mean, roll14_mean, roll30_mean,
    roll7_std, roll14_std, roll30_std,
    cbp_transferred, hhs_discharged,
    cbp_apprehended, net_flow,
    roll_min_30, roll_max_30,
    date: pd.Timestamp,
    feature_cols: list
) -> pd.DataFrame:
    """Build a single feature row matching training feature set."""

    roll_range = roll_max_30 - roll_min_30
    range_pos  = (lag1 - roll_min_30) / roll_range if roll_range > 0 else 0.5

    row = {
        'hhs_care_lag_1'          : lag1,
        'hhs_care_lag_2'          : lag2,
        'hhs_care_lag_3'          : lag3,
        'hhs_care_lag_7'          : lag7,
        'hhs_care_lag_14'         : lag14,
        'hhs_care_roll_mean_7'    : roll7_mean,
        'hhs_care_roll_mean_14'   : roll14_mean,
        'hhs_care_roll_mean_30'   : roll30_mean,
        'hhs_care_roll_std_7'     : roll7_std,
        'hhs_care_roll_std_14'    : roll14_std,
        'hhs_care_roll_std_30'    : roll30_std,
        'hhs_care_roll_min_30'    : roll_min_30,
        'hhs_care_roll_max_30'    : roll_max_30,
        'hhs_care_range_position' : range_pos,
        'cbp_transferred'         : cbp_transferred,
        'cbp_transferred_lag_1'   : cbp_transferred,
        'cbp_transferred_lag_7'   : cbp_transferred,
        'hhs_discharged_lag_1'    : hhs_discharged,
        'hhs_discharged_lag_7'    : hhs_discharged,
        'cbp_apprehended'         : cbp_apprehended,
        'cbp_apprehended_lag_1'   : cbp_apprehended,
        'cbp_apprehended_lag_7'   : cbp_apprehended,
        'net_flow'                : net_flow,
        'net_flow_roll_7'         : net_flow,
        'net_flow_roll_14'        : net_flow,
        'intake_ratio'            : cbp_transferred / (hhs_discharged + 1),
        'intake_ratio_roll_7'     : cbp_transferred / (hhs_discharged + 1),
        'cbp_custody'             : cbp_transferred * 0.3,
        'cbp_apprehended_lag_1'   : cbp_apprehended,
        'is_month_start'          : int(date.is_month_start),
        'is_month_end'            : int(date.is_month_end),
        'season_encoded'          : (date.month % 12) // 3,
        'year'                    : date.year,
        'month'                   : date.month,
        'day_of_week'             : date.dayofweek,
        'day_of_year'             : date.dayofyear,
    }

    # Build dataframe with only the columns the model expects
    feature_df = pd.DataFrame([row])
    for col in feature_cols:
        if col not in feature_df.columns:
            feature_df[col] = 0
    return feature_df[feature_cols]


# ── Sidebar ───────────────────────────────────────────────────
st.sidebar.image(
    "https://www.hhs.gov/sites/default/files/hhs-logo.png",
    width=120
)
st.sidebar.title("UAC Forecasting")
st.sidebar.markdown("**HHS Unaccompanied Children Program**")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    [
        "📊 Overview",
        "🔮 Care Load Forecast",
        "🚪 Discharge Forecast",
        "⚠️ Early Warning System",
        "📈 Model Performance",
        "📋 About"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Model Info**")
st.sidebar.markdown(f"Care Load: `XGBoost` MAE {model_config['careload']['best_mae']}")
st.sidebar.markdown(f"Discharge: `XGBoost` MAE {model_config['discharge']['best_mae']}")
st.sidebar.markdown(f"Data: Jan 2023 – Dec 2025")


# ── Pages ─────────────────────────────────────────────────────
if page == "📊 Overview":
    st.title("🏠 UAC Program — Forecasting Dashboard")
    st.markdown(
        "Predictive analytics for HHS Unaccompanied Alien Children Program "
        "care load and discharge demand."
    )
    st.markdown("---")

    # KPI cards
    latest        = df['hhs_care'].iloc[-1]
    latest_date   = df.index[-1].strftime('%b %d, %Y')
    alert         = get_alert_level(latest)
    alert_color   = get_alert_color(alert)
    alert_emoji   = get_alert_emoji(alert)
    pct_of_stress = (latest / STRESS_THRESHOLD) * 100

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label    = "Latest Care Load",
            value    = f"{int(latest):,}",
            delta    = f"{int(latest - df['hhs_care'].iloc[-2]):+,} vs prev day"
        )
    with col2:
        st.metric(
            label = "Alert Status",
            value = f"{alert_emoji} {alert}"
        )
    with col3:
        st.metric(
            label = "% of Stress Threshold",
            value = f"{pct_of_stress:.1f}%",
            delta = f"Stress at {int(STRESS_THRESHOLD):,}"
        )
    with col4:
        st.metric(
            label = "Historical Mean",
            value = f"{int(HISTORICAL_MEAN):,}",
            delta = f"{int(latest - HISTORICAL_MEAN):+,} vs mean"
        )

    st.markdown("---")

    # Care load trend
    st.subheader("Care Load — Full Historical Trend")
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(df.index, df['hhs_care'],
            color='steelblue', linewidth=1.5, label='HHS Care Load')
    ax.axhline(y=STRESS_THRESHOLD, color='orange',
               linestyle='--', linewidth=1.5,
               label=f'Stress Threshold ({int(STRESS_THRESHOLD):,})')
    ax.axhline(y=CRITICAL_THRESHOLD, color='red',
               linestyle='--', linewidth=1.5,
               label=f'Critical Threshold ({int(CRITICAL_THRESHOLD):,})')
    ax.axvline(x=pd.Timestamp('2025-01-01'), color='purple',
               linestyle=':', linewidth=2, label='Structural Break (Jan 2025)')
    ax.set_ylabel('Children in HHS Care')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Two column summary
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📌 Key Findings")
        st.markdown("""
        - **720** real records, Jan 2023 – Dec 2025
        - **Structural break** Jan 2025 — care load dropped ~66%
        - **XGBoost** beat naive baseline by **9.6%** (MAE 5.48 vs 6.06)
        - **Only 1 model** out of 9 beat the naive baseline
        - **Statistical models failed** due to structural break anchoring
        """)
    with col2:
        st.subheader("🏆 Model Leaderboard")
        display_df = complete_results[['MAE', 'RMSE', 'MAPE']].round(2)
        st.dataframe(display_df.sort_values('MAE'), use_container_width=True)



elif page == "🔮 Care Load Forecast":
    st.title("🔮 Care Load Forecast")
    st.markdown(
        "Enter recent program data to generate a next-day care load prediction."
    )
    st.markdown("---")

    # Recent actual data for reference
    recent = df['hhs_care'].tail(14)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Input Recent Values")
        pred_date    = st.date_input("Prediction Date",
                                      value=df.index[-1].date() + timedelta(days=1))
        lag1         = st.number_input("Yesterday's Care Load (lag 1)",
                                        value=int(df['hhs_care'].iloc[-1]),
                                        min_value=0, max_value=20000)
        lag2         = st.number_input("2 Days Ago (lag 2)",
                                        value=int(df['hhs_care'].iloc[-2]),
                                        min_value=0, max_value=20000)
        lag3         = st.number_input("3 Days Ago (lag 3)",
                                        value=int(df['hhs_care'].iloc[-3]),
                                        min_value=0, max_value=20000)
        lag7         = st.number_input("7 Days Ago (lag 7)",
                                        value=int(df['hhs_care'].iloc[-7]),
                                        min_value=0, max_value=20000)
        lag14        = st.number_input("14 Days Ago (lag 14)",
                                        value=int(df['hhs_care'].iloc[-14]),
                                        min_value=0, max_value=20000)

    with col2:
        st.subheader("Pipeline Inputs")
        cbp_transferred  = st.number_input("CBP Transferred Today",
                                            value=int(df['cbp_transferred'].iloc[-1]),
                                            min_value=0, max_value=5000)
        hhs_discharged   = st.number_input("HHS Discharged Today",
                                            value=int(df['hhs_discharged'].iloc[-1]),
                                            min_value=0, max_value=500)
        cbp_apprehended  = st.number_input("CBP Apprehended Today",
                                            value=int(df['cbp_apprehended'].iloc[-1]),
                                            min_value=0, max_value=10000)

        st.subheader("Rolling Statistics")
        roll7_mean  = st.number_input("7-Day Rolling Mean",
                                       value=float(df['hhs_care'].tail(7).mean()),
                                       min_value=0.0)
        roll14_mean = st.number_input("14-Day Rolling Mean",
                                       value=float(df['hhs_care'].tail(14).mean()),
                                       min_value=0.0)
        roll30_mean = st.number_input("30-Day Rolling Mean",
                                       value=float(df['hhs_care'].tail(30).mean()),
                                       min_value=0.0)
        roll_min_30 = st.number_input("30-Day Rolling Min",
                                       value=float(df['hhs_care'].tail(30).min()),
                                       min_value=0.0)
        roll_max_30 = st.number_input("30-Day Rolling Max",
                                       value=float(df['hhs_care'].tail(30).max()),
                                       min_value=0.0)

    st.markdown("---")

    if st.button("🔮 Generate Care Load Forecast", type="primary"):
        net_flow = cbp_transferred - hhs_discharged

        feature_row = build_feature_row(
            lag1=lag1, lag2=lag2, lag3=lag3,
            lag7=lag7, lag14=lag14,
            roll7_mean=roll7_mean, roll14_mean=roll14_mean,
            roll30_mean=roll30_mean,
            roll7_std=df['hhs_care'].tail(7).std(),
            roll14_std=df['hhs_care'].tail(14).std(),
            roll30_std=df['hhs_care'].tail(30).std(),
            cbp_transferred=cbp_transferred,
            hhs_discharged=hhs_discharged,
            cbp_apprehended=cbp_apprehended,
            net_flow=net_flow,
            roll_min_30=roll_min_30,
            roll_max_30=roll_max_30,
            date=pd.Timestamp(pred_date),
            feature_cols=care_features
        )

        prediction   = careload_model.predict(feature_row)[0]
        alert_level  = get_alert_level(prediction)
        alert_col    = get_alert_color(alert_level)
        alert_em     = get_alert_emoji(alert_level)

        # Result display
        res1, res2, res3 = st.columns(3)
        with res1:
            st.metric("Predicted Care Load",
                      f"{int(prediction):,} children")
        with res2:
            st.metric("Alert Level",
                      f"{alert_em} {alert_level}")
        with res3:
            st.metric("vs Yesterday",
                      f"{int(prediction - lag1):+,} children")

        # Alert message
        if alert_level == 'CRITICAL':
            st.error(f"🔴 CRITICAL: Predicted care load {int(prediction):,} exceeds critical threshold {int(CRITICAL_THRESHOLD):,}. Immediate action required.")
        elif alert_level == 'STRESS':
            st.warning(f"🟡 STRESS: Predicted care load {int(prediction):,} exceeds stress threshold {int(STRESS_THRESHOLD):,}. Prepare additional capacity.")
        else:
            st.success(f"🟢 NORMAL: Predicted care load {int(prediction):,} is within normal operational range.")

        # Forecast chart
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(recent.index, recent.values,
                color='steelblue', linewidth=2,
                marker='o', markersize=4, label='Recent Actual')
        ax.scatter([pd.Timestamp(pred_date)], [prediction],
                   color=alert_col, s=150, zorder=5,
                   label=f'Prediction: {int(prediction):,}')
        ax.axhline(y=STRESS_THRESHOLD, color='orange',
                   linestyle='--', linewidth=1,
                   label=f'Stress ({int(STRESS_THRESHOLD):,})')
        ax.axhline(y=CRITICAL_THRESHOLD, color='red',
                   linestyle='--', linewidth=1,
                   label=f'Critical ({int(CRITICAL_THRESHOLD):,})')
        ax.set_title(f'Care Load Forecast for {pred_date}', fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()


elif page == "🚪 Discharge Forecast":
    st.title("🚪 Discharge Demand Forecast")
    st.markdown(
        "Predict next-day discharge volume to plan placement capacity."
    )
    st.markdown("---")

    st.subheader("Input Values")
    col1, col2 = st.columns(2)

    with col1:
        dis_date        = st.date_input("Prediction Date",
                                         value=df.index[-1].date() + timedelta(days=1))
        care_lag1       = st.number_input("Yesterday's Care Load",
                                           value=int(df['hhs_care'].iloc[-1]),
                                           min_value=0, max_value=20000)
        care_lag2       = st.number_input("Care Load 2 Days Ago",
                                           value=int(df['hhs_care'].iloc[-2]),
                                           min_value=0, max_value=20000)
        cbp_trans       = st.number_input("CBP Transferred Today",
                                           value=int(df['cbp_transferred'].iloc[-1]),
                                           min_value=0, max_value=5000)
    with col2:
        cbp_appre       = st.number_input("CBP Apprehended Today",
                                           value=int(df['cbp_apprehended'].iloc[-1]),
                                           min_value=0, max_value=10000)
        roll7           = st.number_input("7-Day Care Load Rolling Mean",
                                           value=float(df['hhs_care'].tail(7).mean()),
                                           min_value=0.0)
        roll30_mn       = st.number_input("30-Day Care Load Rolling Mean",
                                           value=float(df['hhs_care'].tail(30).mean()),
                                           min_value=0.0)
        roll30_min_val  = st.number_input("30-Day Rolling Min",
                                           value=float(df['hhs_care'].tail(30).min()),
                                           min_value=0.0)
        roll30_max_val  = st.number_input("30-Day Rolling Max",
                                           value=float(df['hhs_care'].tail(30).max()),
                                           min_value=0.0)

    st.markdown("---")

    if st.button("🚪 Generate Discharge Forecast", type="primary"):
        dis_date_ts = pd.Timestamp(dis_date)
        net_flow_d  = cbp_trans - care_lag1 * 0.04

        dis_row = {
            'hhs_care_lag_1'         : care_lag1,
            'hhs_care_lag_2'         : care_lag2,
            'hhs_care_lag_3'         : care_lag2,
            'hhs_care_lag_7'         : care_lag1,
            'hhs_care_lag_14'        : care_lag1,
            'hhs_care_roll_mean_7'   : roll7,
            'hhs_care_roll_mean_14'  : roll7,
            'hhs_care_roll_mean_30'  : roll30_mn,
            'hhs_care_roll_std_7'    : df['hhs_care'].tail(7).std(),
            'hhs_care_roll_std_14'   : df['hhs_care'].tail(14).std(),
            'hhs_care_roll_std_30'   : df['hhs_care'].tail(30).std(),
            'hhs_care_roll_min_30'   : roll30_min_val,
            'hhs_care_roll_max_30'   : roll30_max_val,
            'hhs_care_range_position': (care_lag1 - roll30_min_val) / max(roll30_max_val - roll30_min_val, 1),
            'cbp_transferred'        : cbp_trans,
            'cbp_transferred_lag_1'  : cbp_trans,
            'cbp_transferred_lag_7'  : cbp_trans,
            'cbp_apprehended'        : cbp_appre,
            'cbp_apprehended_lag_1'  : cbp_appre,
            'cbp_apprehended_lag_7'  : cbp_appre,
            'net_flow'               : net_flow_d,
            'net_flow_roll_7'        : net_flow_d,
            'net_flow_roll_14'       : net_flow_d,
            'intake_ratio'           : cbp_trans / (care_lag1 * 0.04 + 1),
            'intake_ratio_roll_7'    : cbp_trans / (care_lag1 * 0.04 + 1),
            'cbp_custody'            : cbp_trans * 0.3,
            'is_month_start'         : int(dis_date_ts.is_month_start),
            'is_month_end'           : int(dis_date_ts.is_month_end),
            'season_encoded'         : (dis_date_ts.month % 12) // 3,
            'year'                   : dis_date_ts.year,
            'month'                  : dis_date_ts.month,
            'day_of_week'            : dis_date_ts.dayofweek,
            'day_of_year'            : dis_date_ts.dayofyear,
        }

        dis_df = pd.DataFrame([dis_row])
        for col in discharge_features:
            if col not in dis_df.columns:
                dis_df[col] = 0
        dis_df = dis_df[discharge_features]

        dis_pred = max(0, discharge_model.predict(dis_df)[0])

        r1, r2, r3 = st.columns(3)
        with r1:
            st.metric("Predicted Discharges", f"{dis_pred:.1f} children/day")
        with r2:
            st.metric("Recent Avg Discharge",
                      f"{df['hhs_discharged'].tail(7).mean():.1f} children/day")
        with r3:
            st.metric("Model MAE", "± 0.63 children/day")

        st.success(
            f"Predicted **{dis_pred:.1f} discharges** on {dis_date}. "
            f"Plan placement capacity accordingly."
        )

        # Discharge trend
        fig, ax = plt.subplots(figsize=(12, 4))
        recent_d = df['hhs_discharged'].tail(30)
        ax.bar(recent_d.index, recent_d.values,
               color='steelblue', alpha=0.7, label='Recent Discharges')
        ax.axhline(y=dis_pred, color='red', linestyle='--',
                   linewidth=2, label=f'Prediction: {dis_pred:.1f}')
        ax.axhline(y=recent_d.mean(), color='orange', linestyle=':',
                   linewidth=1.5, label=f'30-day avg: {recent_d.mean():.1f}')
        ax.set_title(f'Discharge Forecast for {dis_date}', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()



elif page == "⚠️ Early Warning System":
    st.title("⚠️ Early Warning System")
    st.markdown("Capacity stress monitoring based on historical thresholds.")
    st.markdown("---")

    # Threshold cards
    c1, c2, c3 = st.columns(3)
    with c1:
        st.info(f"🟢 **Normal**\nBelow {int(STRESS_THRESHOLD):,}")
    with c2:
        st.warning(f"🟡 **Stress**\n{int(STRESS_THRESHOLD):,} – {int(CRITICAL_THRESHOLD):,}")
    with c3:
        st.error(f"🔴 **Critical**\nAbove {int(CRITICAL_THRESHOLD):,}")

    st.markdown("---")

    # Full trend with zones
    st.subheader("Historical Care Load with Alert Zones")
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.fill_between(df.index, CRITICAL_THRESHOLD,
                    df['hhs_care'].max() * 1.05,
                    alpha=0.15, color='red', label='Critical Zone')
    ax.fill_between(df.index, STRESS_THRESHOLD, CRITICAL_THRESHOLD,
                    alpha=0.15, color='orange', label='Stress Zone')
    ax.fill_between(df.index, 0, STRESS_THRESHOLD,
                    alpha=0.1, color='green', label='Normal Zone')
    ax.plot(df.index, df['hhs_care'],
            color='steelblue', linewidth=1.5, label='Care Load')
    ax.axvline(x=pd.Timestamp('2025-01-01'), color='purple',
               linestyle=':', linewidth=2, label='Structural Break')
    ax.set_ylabel('Children in HHS Care')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Alert history — last 90 days
    st.subheader("Alert History — Last 90 Days")
    last90 = df['hhs_care'].tail(90)
    alerts = [get_alert_level(v) for v in last90.values]
    colors = [get_alert_color(a) for a in alerts]

    from collections import Counter
    counts = Counter(alerts)
    a1, a2, a3 = st.columns(3)
    with a1:
        st.metric("🟢 Normal Days", counts.get('NORMAL', 0))
    with a2:
        st.metric("🟡 Stress Days", counts.get('STRESS', 0))
    with a3:
        st.metric("🔴 Critical Days", counts.get('CRITICAL', 0))

    fig2, ax2 = plt.subplots(figsize=(14, 3))
    alert_num = [{'NORMAL': 1, 'STRESS': 2, 'CRITICAL': 3}[a] for a in alerts]
    ax2.bar(last90.index, alert_num, color=colors, alpha=0.8, width=1)
    ax2.set_yticks([1, 2, 3])
    ax2.set_yticklabels(['NORMAL', 'STRESS', 'CRITICAL'])
    ax2.set_title('Daily Alert Levels — Last 90 Days', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()

    st.markdown("---")
    st.subheader("Threshold Configuration")
    st.markdown(f"""
    | Level    | Threshold | Basis |
    |----------|-----------|-------|
    | Normal   | Below {int(STRESS_THRESHOLD):,} | Below 75th percentile |
    | Stress   | {int(STRESS_THRESHOLD):,} – {int(CRITICAL_THRESHOLD):,} | 75th – 90th percentile |
    | Critical | Above {int(CRITICAL_THRESHOLD):,} | Above 90th percentile |

    Thresholds calculated from full historical data (Jan 2023 – Dec 2025).
    Historical mean: **{int(HISTORICAL_MEAN):,}** children.
    """)



elif page == "📈 Model Performance":
    st.title("📈 Model Performance")
    st.markdown("Complete evaluation results across all models and approaches.")
    st.markdown("---")

    # Winner callout
    st.success(
        "🏆 **Best Model: XGBoost (Recent Window)** — "
        "MAE 5.48 children/day — 9.6% improvement over naive baseline"
    )

    # Complete leaderboard
    st.subheader("Complete Leaderboard")
    display = complete_results[['MAE', 'RMSE', 'MAPE']].round(2).sort_values('MAE')
    st.dataframe(display, use_container_width=True)

    st.markdown("---")

    # Escalation chart
    st.subheader("Model Escalation — From Simple to Complex")
    st.image(str(FIGURES_DIR / '07_escalation_story.png'),
             use_container_width=True)

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Care Load — Deep Dive")
        st.image(str(FIGURES_DIR / '07_careload_deep_dive.png'),
                 use_container_width=True)
    with col2:
        st.subheader("Discharge — Deep Dive")
        st.image(str(FIGURES_DIR / '07_discharge_deep_dive.png'),
                 use_container_width=True)

    st.markdown("---")
    st.subheader("Early Warning System Validation")
    st.image(str(FIGURES_DIR / '07_early_warning_system.png'),
             use_container_width=True)

    st.markdown("---")
    st.subheader("Key Metrics Summary")
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Care Load MAE", "5.48", delta="-0.58 vs naive")
    with m2:
        st.metric("Care Load MAPE", "0.23%")
    with m3:
        st.metric("Discharge MAE", "0.63")
    with m4:
        st.metric("Models Tested", "9")



elif page == "📋 About":
    st.title("📋 About This Project")
    st.markdown("---")

    st.markdown("""
    ## Predictive Forecasting of Care Load & Placement Demand
    ### HHS Unaccompanied Alien Children Program

    ---

    ### Project Summary
    This dashboard presents machine learning models trained on HHS UAC Program
    data from January 2023 to December 2025 to forecast daily care load and
    discharge demand.

    ---

    ### Dataset
    | Property | Value |
    |---|---|
    | Source | HHS UAC Program public data |
    | Records | 720 real observations |
    | Date range | Jan 2023 – Dec 2025 |
    | After interpolation | 1,075 rows (355 missing dates filled) |
    | Target 1 | `hhs_care` — children in HHS custody daily |
    | Target 2 | `hhs_discharged` — daily discharges |

    ---

    ### Key Finding — Structural Break
    In January 2025 care load dropped from ~6,500 to ~2,200 — a 66% reduction.
    This single event caused all statistical models and full-dataset ML models to fail.
    The solution was **recent window retraining** — training only on data from
    June 2024 onwards for care load, and March 2025 onwards for discharge.

    ---

    ### Model Results
    | Model | MAE | vs Naive |
    |---|---|---|
    | XGBoost (Recent) | 5.48 | ✅ Beat naive by 9.6% |
    | Naive Persistence | 6.06 | — baseline |
    | Random Forest (Recent) | 6.54 | ❌ |
    | Statistical Models | 86–433 | ❌ Structural break |

    ---

    ### Tech Stack
    `Python` · `XGBoost` · `scikit-learn` · `pandas` · `Streamlit`""")

   