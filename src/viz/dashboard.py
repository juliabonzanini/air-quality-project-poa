"""
Dashboard Interativo
Projeto: Modelo Preditivo de Qualidade do Ar – Porto Alegre
Autora: Júlia Valandro Bonzanini
Revisão: Outubro/2025
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Dict
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.graph_objects as go
import datetime as dt
import pytz
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ==========================
# CONFIG
# ==========================

THIS_FILE = Path(__file__).resolve()
ROOT_DIR = THIS_FILE.parents[2]
st.sidebar.info(f"📂 Diretório base detectado:\n`{ROOT_DIR}`")

class Config:
    DATA_DIR = ROOT_DIR / "data" / "processed"
    MODELS_DIR = ROOT_DIR / "models"
    REPORTS_DIR = ROOT_DIR / "reports" / "model_refined"

    FEATURES_TRAIN = DATA_DIR / "air_quality_features_pro.csv"
    FEATURES_FULL  = DATA_DIR / "air_quality_features_full.csv"

    ARTIFACTS = MODELS_DIR / "meta_ensemble_pm10_oof_stack.joblib"
    QUANTILES = MODELS_DIR / "quantile_models.joblib"

    REFINED_JSON = REPORTS_DIR / "refined_model_results.json"
    SEASONAL_CSV = REPORTS_DIR / "seasonal_calibration.csv"
    ISOTONIC_CSV = REPORTS_DIR / "meta_isotonic_metrics.csv"

    YEARLY_METRICS = REPORTS_DIR / "metrics_by_year.csv"
    YEARLY_PNG     = REPORTS_DIR / "yearly_stability.png"

    DRIFT_CSV = REPORTS_DIR / "feature_drift_accumulated.csv"
    DRIFT_PNG = REPORTS_DIR / "feature_drift_accumulated.png"

    CMP_R2_PNG   = REPORTS_DIR / "compare_r2_stability.png"
    CMP_RMSE_PNG = REPORTS_DIR / "compare_rmse_stability.png"
    CMP_SUMMARY  = REPORTS_DIR / "compare_stability_summary.csv"
    CMP_TXT      = REPORTS_DIR / "compare_stability_summary.txt"

    TARGET = "PM10_Canoas"
    WINTER_MONTHS = (6, 7, 8)

st.set_page_config(page_title="Air Quality Dashboard – POA", layout="wide")

# ==========================
# HELPERS
# ==========================

@st.cache_data
def load_json(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

@st.cache_data
def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)

@st.cache_resource
def load_artifacts() -> Dict:
    return joblib.load(Config.ARTIFACTS)

@st.cache_resource
def load_quantiles() -> Dict:
    return joblib.load(Config.QUANTILES)

def _season_mask(dt: pd.Series) -> np.ndarray:
    months = pd.to_datetime(dt).dt.month.values
    return np.isin(months, Config.WINTER_MONTHS)

def _pad_or_trim(X: np.ndarray, expected_cols: int) -> np.ndarray:
    cur = X.shape[1]
    if cur < expected_cols:
        pad = np.zeros((X.shape[0], expected_cols - cur))
        return np.hstack([X, pad])
    elif cur > expected_cols:
        return X[:, :expected_cols]
    return X

def _build_meta_features_for_inference(df_block: pd.DataFrame, preds_base: np.ndarray) -> np.ndarray:
    parts = [preds_base]
    day = pd.to_datetime(df_block["datetime"]).dt.dayofyear.to_numpy()
    parts.append(np.sin(2 * np.pi * day / 365.0)[:, None])
    parts.append(np.cos(2 * np.pi * day / 365.0)[:, None])
    parts.append(_season_mask(df_block["datetime"]).astype(int)[:, None])

    for c in [
        "PM10_Canoas_roll_mean_3", "PM10_Canoas_roll_mean_7",
        "pm10_ratio_3_30", "pm10_diff_3_30"
    ]:
        if c in df_block.columns:
            vec = df_block[c].astype(float).ffill().fillna(0.0).to_numpy()
            parts.append(vec[:, None])
        else:
            parts.append(np.zeros((len(df_block), 1)))

    meta = np.hstack(parts)
    meta = np.nan_to_num(meta, nan=0.0, posinf=0.0, neginf=0.0)
    return meta

def classify_iqar(pm10_value: float) -> str:
    """Classifica a qualidade do ar com base no valor de PM10 (µg/m³)"""
    if pm10_value <= 45:
        return "BOA"
    elif pm10_value <= 100:
        return "MODERADA"
    elif pm10_value <= 150:
        return "RUIM"
    elif pm10_value <= 250:
        return "MUITO RUIM"
    else:
        return "PÉSSIMA"

def color_iqar(class_name: str) -> str:
    """Retorna cor hex para cada classe IQAr"""
    colors = {
        "BOA": "#2ECC71",        # verde
        "MODERADA": "#F1C40F",   # amarelo
        "RUIM": "#E67E22",       # laranja
        "MUITO RUIM": "#E74C3C", # vermelho
        "PÉSSIMA": "#8E44AD"     # roxo
    }
    return colors.get(class_name.upper(), "#BDC3C7")

# ==========================
# DASHBOARD
# ==========================

st.title("📊 Dashboard – Modelo Preditivo de Qualidade do Ar (POA)")
st.caption("Transparência temporal • Sazonalidade de inverno • Meta-ensemble calibrado")

# ---------- OVERVIEW ----------
st.header("🧠 Overview do Modelo")
if Config.REFINED_JSON.exists():
    info = load_json(Config.REFINED_JSON)
    col1, col2, col3 = st.columns(3)
    col1.metric("R² (global, isotônico)", f"{float(info.get('global_r2_after_isotonic', np.nan)):.3f}")
    col2.metric("Amostras", f"{info.get('n_samples', '—')}")
    col3.metric("Features", f"{info.get('n_features', '—')}")
else:
    st.warning("Arquivo refined_model_results.json não encontrado.")

st.divider()

# ---------- ESTABILIDADE ----------
st.header("📈 Estabilidade Temporal")
cols = st.columns(2)
if Config.YEARLY_PNG.exists():
    cols[0].image(str(Config.YEARLY_PNG), caption="Estabilidade por Ano")
if Config.YEARLY_METRICS.exists():
    yearly = load_csv(Config.YEARLY_METRICS)
    cols[1].dataframe(yearly)

st.divider()

# ---------- DRIFT ----------
st.header("🌦️ Drift Acumulado de Features")
cols = st.columns(2)
if Config.DRIFT_PNG.exists():
    cols[0].image(str(Config.DRIFT_PNG), caption="Top 10 – Drift Acumulado")
if Config.DRIFT_CSV.exists():
    drift = load_csv(Config.DRIFT_CSV)
    cols[1].dataframe(drift.head(15))

st.divider()

# ---------- COMPARATIVO ----------
st.header("🔁 Comparativo: Estático vs Incremental")
cols = st.columns(2)
if Config.CMP_R2_PNG.exists():
    cols[0].image(str(Config.CMP_R2_PNG), caption="R² Estático vs Incremental")
if Config.CMP_RMSE_PNG.exists():
    cols[1].image(str(Config.CMP_RMSE_PNG), caption="RMSE Estático vs Incremental")
if Config.CMP_SUMMARY.exists():
    comp = load_csv(Config.CMP_SUMMARY)
    st.dataframe(comp)
if Config.CMP_TXT.exists():
    st.code(Path(Config.CMP_TXT).read_text(encoding="utf-8"))

st.divider()

# ---------- PREVISÃO ----------
st.header("🧭 Previsão e Calibração (customizável)")

if not Config.ARTIFACTS.exists() or not Config.QUANTILES.exists():
    st.error("Artefatos de modelo não encontrados.")
else:
    artifacts = load_artifacts()
    quantiles = load_quantiles()
    feats = artifacts["feature_names"]

    mode = st.radio("Modo de previsão:",
                    ["Previsão simples (estática)", "Previsão recursiva (dinâmica)"],
                    horizontal=True)
    drift_pct = st.slider("Ajuste percentual nas exógenas (%):", -10, 10, 0, 1)
    horizon = st.slider("Horizonte de previsão (dias):", 1, 7, 7)
    randomize_exog = st.checkbox("Adicionar variação aleatória (~5%) nas exógenas", value=True)

    tz = pytz.timezone("America/Sao_Paulo")
    df_full = load_csv(Config.FEATURES_FULL)
    df_full["datetime"] = pd.to_datetime(df_full["datetime"])
    min_date = df_full["datetime"].min().date()
    today = dt.datetime.now(tz).date()
    max_future = today + dt.timedelta(days=180)

    # 📆 Médias semanais históricas
    df_full["weekofyear"] = pd.to_datetime(df_full["datetime"]).dt.isocalendar().week
    df_weekly_means = (
        df_full.groupby("weekofyear")
        .mean(numeric_only=True)
        .rename_axis("weekofyear")
    )

    st.markdown("### 📅 Escolha da data inicial")
    selected_date = st.date_input(
        "Data inicial da previsão",
        value=today,
        min_value=min_date,
        max_value=max_future,
        help="Selecione a data inicial (até 6 meses no futuro)."
    )

    if st.button(f"▶️ Rodar previsão de {horizon} dias"):
        start_dt = pd.Timestamp(selected_date)
        fut_dates = pd.date_range(start_dt, periods=horizon, freq="D")

        hist = df_full.copy()
        preds_list = []

        for fdate in fut_dates:
            hist["datetime"] = pd.to_datetime(hist["datetime"], errors="coerce").dt.tz_localize(None)
            fdate_naive = pd.Timestamp(fdate).tz_localize(None)
            ref = hist[hist["datetime"] <= fdate_naive]
            row = ref.iloc[[-1]].copy() if not ref.empty else hist.iloc[[-1]].copy()

            # Substitui exógenas pelas médias semanais históricas
            week = pd.Timestamp(fdate).isocalendar().week
            week = min(week, df_weekly_means.index.max())
            mean_vals = df_weekly_means.loc[week]
            for c in mean_vals.index:
                if c in row.columns and c not in ["PM10_Canoas", "datetime"]:
                    row[c] = mean_vals[c]

            # Ruído leve
            if randomize_exog:
                noise = np.random.normal(1.0, 0.05, size=len(mean_vals))
                for c, n in zip(mean_vals.index, noise):
                    if c in row.columns and c not in ["PM10_Canoas", "datetime"]:
                        row[c] *= n

            # Features temporais
            row["datetime"] = fdate
            row["month"] = fdate.month
            row["weekday"] = fdate.weekday()
            row["dayofyear"] = fdate.timetuple().tm_yday
            row["sin_dayofyear"] = np.sin(2 * np.pi * row["dayofyear"] / 365.0)
            row["cos_dayofyear"] = np.cos(2 * np.pi * row["dayofyear"] / 365.0)
            row["is_weekend"] = int(fdate.weekday() >= 5)
            row["is_winter"] = int(fdate.month in Config.WINTER_MONTHS)
            row["year"] = fdate.year

            # Drift ajustável
            exog_cols = [c for c in feats if not c.startswith("PM10") and c != "datetime"]
            if abs(drift_pct) > 0:
                row[exog_cols] = row[exog_cols] * (1.0 + drift_pct / 100.0)

            row = row.reindex(columns=feats, fill_value=0.0)
            row = row.apply(pd.to_numeric, errors="coerce").ffill().fillna(0.0)

            # Previsão base
            X = row.to_numpy(dtype=float)
            base_preds = np.column_stack([
                m.predict(_pad_or_trim(X, m.n_features_in_), feature_name="auto")
                if "feature_name" in m.predict.__code__.co_varnames else
                m.predict(_pad_or_trim(X, m.n_features_in_))
                for _, m in artifacts["base_models"].items()
            ])

            meta_feats = _build_meta_features_for_inference(row.assign(datetime=row["datetime"]), base_preds)

            exp_w = getattr(artifacts["meta_winter"], "n_features_in_", meta_feats.shape[1])
            exp_nw = getattr(artifacts["meta_nonwinter"], "n_features_in_", meta_feats.shape[1])
            meta_feats_w = _pad_or_trim(meta_feats, exp_w)
            meta_feats_nw = _pad_or_trim(meta_feats, exp_nw)

            wmask = _season_mask(row["datetime"])
            if wmask.any():
                y_hat = artifacts["meta_winter"].predict(meta_feats_w)
                pm10 = artifacts["iso_winter"].predict(y_hat)
            else:
                y_hat = artifacts["meta_nonwinter"].predict(meta_feats_nw)
                pm10 = artifacts["iso_nonwinter"].predict(y_hat)

            pm10 = np.nan_to_num(pm10, nan=0.0)
            preds_list.append(pd.Series({
                "datetime": fdate,
                "pm10": float(pm10[0]),
                "q10": float(pm10[0]) * 0.9,
                "q90": float(pm10[0]) * 1.1,
                "IQAr_Class": classify_iqar(float(pm10[0]))
            }))

            if "recursiva" in mode.lower():
                hist.loc[len(hist)] = row.iloc[0]
                hist.at[len(hist)-1, "PM10_Canoas"] = float(pm10[0])

        preds = pd.DataFrame(preds_list)

        # === Gráfico ===
        colors = preds["IQAr_Class"].map(color_iqar)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=preds["datetime"], y=preds["pm10"],
            mode="lines+markers", name="Previsão calibrada",
            marker=dict(color=colors, size=10),
            line=dict(color="rgba(0,0,0,0.3)")
        ))
        fig.add_trace(go.Scatter(
            x=list(preds["datetime"]) + list(preds["datetime"][::-1]),
            y=list(preds["q90"]) + list(preds["q10"][::-1]),
            fill="toself", fillcolor="rgba(31,119,180,0.15)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip", name="Faixa p10–p90"
        ))
        fig.update_layout(
            title=f"{mode} – PM₁₀ próximos {horizon} dias",
            xaxis_title="Data", yaxis_title="PM₁₀ (µg/m³)",
            template="plotly_white",
            autosize=True,
            margin=dict(l=40, r=40, t=60, b=40)
        )
        st.plotly_chart(
            fig,
            config={
                "responsive": True,
                "displaylogo": False,
                "scrollZoom": True,
                "modeBarButtonsToRemove": ["lasso2d", "select2d"]
            }
        )

        # Legenda IQAr
        st.markdown("""
        <div style='text-align:center;'>
        🟢 <b>BOA</b> &nbsp;&nbsp;
        🟡 <b>MODERADA</b> &nbsp;&nbsp;
        🟠 <b>RUIM</b> &nbsp;&nbsp;
        🔴 <b>MUITO RUIM</b> &nbsp;&nbsp;
        🟣 <b>PÉSSIMA</b>
        </div>
        """, unsafe_allow_html=True)

        # === Resumo ===
        avg, low, high = preds["pm10"].mean(), preds["pm10"].min(), preds["pm10"].max()
        st.success(
            f"**Resumo das previsões:** média = {avg:.2f} µg/m³ | mínimo = {low:.2f} | "
            f"máximo = {high:.2f} | amplitude = {high - low:.2f}"
        )

        # === Tabela + Download ===
        st.dataframe(preds[["datetime", "pm10", "IQAr_Class", "q10", "q90"]]
                     .style.format({"pm10": "{:.2f}", "q10": "{:.2f}", "q90": "{:.2f}"}))
        st.download_button(
            "📥 Baixar previsões (CSV)",
            data=preds.to_csv(index=False).encode("utf-8"),
            file_name=f"forecast_{horizon}d_{selected_date}.csv"
        )

st.divider()

# ---------- DOWNLOADS ----------
st.header("📦 Downloads Rápidos")
downloads = [Config.REFINED_JSON, Config.YEARLY_METRICS, Config.DRIFT_CSV, Config.CMP_SUMMARY]
cols = st.columns(4)
for i, p in enumerate(downloads):
    if p.exists():
        with open(p, "rb") as f:
            cols[i].download_button(f"Baixar {p.name}", data=f, file_name=p.name)
    else:
        cols[i].warning(f"{p.name} indisponível.")
