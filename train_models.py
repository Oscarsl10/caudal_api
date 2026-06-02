"""
train_models.py
---------------
Script independiente que entrena los cuatro modelos sobre la serie completa,
evalúa cada uno en el conjunto test (hold-out de 60 días) y serializa
todos los artefactos necesarios para la API.

Ejecutar una sola vez antes de levantar la API:
    python train_models.py

Artefactos generados en MODEL_STORE_PATH:
    ensemble_model.pkl          → GradientBoostingRegressor (Entrenado en Train)
    ensemble_feature_cols.pkl   → lista de columnas de features
    scaler_features.pkl         → MinMaxScaler ajustado sobre X_train
    test_metrics.pkl            → dict con RMSE, MAE, MAPE, NSE, R², MDA por modelo
    serie_historica.pkl         → pd.Series con la serie completa (train + test)
"""

import os
import pickle
import warnings
from io import StringIO
from contextlib import redirect_stdout

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Configuración
# ─────────────────────────────────────────────────────────────────────────────
DATA_PATH = os.environ.get("DATA_PATH", "./data")
MODEL_STORE_PATH = os.environ.get("MODEL_STORE_PATH", "./models_store")
CSV_FILE = os.path.join(DATA_PATH, "caudal_limpio_diario.csv")

HORIZONTE_TEST = 60   # días reservados para evaluación out-of-sample
LAGS = [1, 7, 14, 30]

SARIMA_ORDER = (1, 1, 1)
SARIMA_SEASONAL_ORDER = (1, 0, 0, 30)

HW_TREND = "add"
HW_SEASONAL = "add"
HW_SEASONAL_PERIODS = 30

PROPHET_CONFIG = {
    "yearly_seasonality": True,
    "weekly_seasonality": False,
    "daily_seasonality": False,
    "changepoint_prior_scale": 0.10,
    "seasonality_prior_scale": 5.0,
    "seasonality_mode": "additive",
    "interval_width": 0.80,
}

GB_PARAMS = {
    "n_estimators": 200,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "random_state": 42,
    "verbose": 0,
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_pkl(obj, name: str) -> None:
    path = os.path.join(MODEL_STORE_PATH, name)
    with open(path, "wb") as fh:
        pickle.dump(obj, fh)
    print(f"    ✓ {name}")


def nash_sutcliffe_efficiency(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    num = np.sum((y_true - y_pred) ** 2)
    den = np.sum((y_true - np.mean(y_true)) ** 2)
    if den == 0:
        return 0.0
    return float(1 - num / den)


def mean_directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    true_diff = np.sign(y_true[1:] - y_true[:-1])
    pred_diff = np.sign(y_pred[1:] - y_pred[:-1])
    return float(np.mean(true_diff == pred_diff)) * 100  # en porcentaje


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    with np.errstate(divide="ignore", invalid="ignore"):
        raw_mape = np.abs((y_true - y_pred) / y_true) * 100
        mape = float(np.nanmean(raw_mape[np.isfinite(raw_mape)]))
    nse = nash_sutcliffe_efficiency(y_true, y_pred)
    r2 = float(r2_score(y_true, y_pred))
    mda = mean_directional_accuracy(y_true, y_pred)
    return {"rmse": round(rmse, 4), "mae": round(mae, 4), "mape": round(mape, 4),
            "nse": round(nse, 4), "r2": round(r2, 4), "mda": round(mda, 2)}


def build_lag_features(serie: pd.Series) -> pd.DataFrame:
    """
    Construye las características temporales replicando la ingeniería del Notebook
    """
    df = pd.DataFrame(index=serie.index)
    for lag in LAGS:
        df[f"lag_{lag}"] = serie.shift(lag)
    
    # El shift(1) evita el data leakage en las ventanas móviles
    df["rolling_mean_7"] = serie.shift(1).rolling(7).mean()
    df["rolling_mean_14"] = serie.shift(1).rolling(14).mean()
    df["rolling_std_7"] = serie.shift(1).rolling(7).std()
    df["rolling_std_14"] = serie.shift(1).rolling(14).std()
    
    # Inyección de las variables estacionales deterministas del índice
    df["day_of_year"] = df.index.dayofyear
    df["month"] = df.index.month
    df["day_of_week"] = df.index.dayofweek
    
    df["target"] = serie.values
    return df.dropna()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    os.makedirs(MODEL_STORE_PATH, exist_ok=True)

    # ── 1. Cargar serie limpia ──────────────────────────────────────────────
    print("\n📂 Cargando serie histórica...")
    df = pd.read_csv(CSV_FILE, index_col="Fecha", parse_dates=True)
    df = df.asfreq("D")
    serie = df["Caudal"]
    print(f"   {serie.index.min().date()} → {serie.index.max().date()} | {len(serie)} registros")

    train = serie.iloc[:-HORIZONTE_TEST]
    test = serie.iloc[-HORIZONTE_TEST:]
    y_test = test.values

    test_metrics: dict[str, dict[str, float]] = {}

    # ── 2. SARIMA ──────────────────────────────────────────────────────────
    print("\n🔄 Entrenando SARIMA...")
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    m_sarima = SARIMAX(
        train, order=SARIMA_ORDER, seasonal_order=SARIMA_SEASONAL_ORDER,
        enforce_stationarity=False, enforce_invertibility=False,
    )
    r_sarima = m_sarima.fit(disp=False, maxiter=300)
    pred_sarima = r_sarima.forecast(steps=HORIZONTE_TEST).values
    test_metrics["SARIMA"] = compute_metrics(y_test, pred_sarima)
    print(f"   RMSE={test_metrics['SARIMA']['rmse']:.4f}  MAE={test_metrics['SARIMA']['mae']:.4f}")

    # ── 3. Holt-Winters ────────────────────────────────────────────────────
    print("\n🔄 Entrenando Holt-Winters...")
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    m_hw = ExponentialSmoothing(
        train, trend=HW_TREND, seasonal=HW_SEASONAL,
        seasonal_periods=HW_SEASONAL_PERIODS, initialization_method="estimated",
    )
    r_hw = m_hw.fit(optimized=True, remove_bias=True)
    pred_hw = r_hw.forecast(steps=HORIZONTE_TEST).values
    test_metrics["Holt-Winters"] = compute_metrics(y_test, pred_hw)
    print(f"   RMSE={test_metrics['Holt-Winters']['rmse']:.4f}  MAE={test_metrics['Holt-Winters']['mae']:.4f}")

    # ── 4. Prophet ─────────────────────────────────────────────────────────
    print("\n🔄 Entrenando Prophet...")
    try:
        from prophet import Prophet

        df_pr = train.reset_index().rename(columns={"Fecha": "ds", "Caudal": "y"})
        if "ds" not in df_pr.columns:
            df_pr.columns = ["ds", "y"]

        m_prophet = Prophet(**PROPHET_CONFIG)
        with redirect_stdout(StringIO()):
            m_prophet.fit(df_pr)

        future = m_prophet.make_future_dataframe(periods=HORIZONTE_TEST, freq="D")
        forecast_pr = m_prophet.predict(future)
        pred_prophet = forecast_pr.set_index("ds")["yhat"].iloc[-HORIZONTE_TEST:].values
        test_metrics["Prophet"] = compute_metrics(y_test, pred_prophet)
        print(f"   RMSE={test_metrics['Prophet']['rmse']:.4f}  MAE={test_metrics['Prophet']['mae']:.4f}")

    except Exception as prophet_err:
        print(f"   WARNING Prophet fallo: {prophet_err}")
        test_metrics["Prophet"] = {
            "rmse": float("inf"), "mae": float("inf"), "mape": float("inf"),
            "nse": float("-inf"), "r2": float("-inf"), "mda": 0.0,
        }

    # ── 5. Ensamble (Gradient Boosting) ────────────────────────────────────
    print("\n🔄 Entrenando Ensamble (Gradient Boosting)...")
    df_features = build_lag_features(train)
    feature_cols = [c for c in df_features.columns if c != "target"]

    X_train_gb = df_features[feature_cols].values
    y_train_gb = df_features["target"].values

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train_gb)

    gb_model = GradientBoostingRegressor(**GB_PARAMS)
    gb_model.fit(X_train_scaled, y_train_gb)

    # Evaluación One-Step-Ahead en Test usando historia real (Métricas del Notebook)
    df_features_full_eval = build_lag_features(serie)
    
    X_test_notebook = df_features_full_eval.loc[test.index, feature_cols].values
    y_test_notebook = df_features_full_eval.loc[test.index, "target"].values
    
    X_test_scaled = scaler.transform(X_test_notebook)
    pred_ensemble = gb_model.predict(X_test_scaled)
    
    test_metrics["Ensamble"] = compute_metrics(y_test_notebook, pred_ensemble)
    print(f"   RMSE={test_metrics['Ensamble']['rmse']:.4f}  MAE={test_metrics['Ensamble']['mae']:.4f}")

    # ── 6. Serialización Limpia (Conserva el modelo original de Train) ──────
    print("\n💾 Serializando artefactos científicos del Notebook...")
    save_pkl(gb_model, "ensemble_model.pkl")
    save_pkl(feature_cols, "ensemble_feature_cols.pkl")
    save_pkl(scaler, "scaler_features.pkl")
    save_pkl(test_metrics, "test_metrics.pkl")
    save_pkl(serie, "serie_historica.pkl")

    # ── 7. Resumen final ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("🏆 RANKING POR RMSE EN TEST:")
    for i, (nombre, m) in enumerate(
        sorted(test_metrics.items(), key=lambda x: x[1]["rmse"]), 1
    ):
        print(f"   {i}. {nombre:20s}   RMSE={m['rmse']:.4f}   NSE={m['nse']:.4f}")
    print(f"\n✅ Todos los artefactos en: {MODEL_STORE_PATH}")


if __name__ == "__main__":
    main()