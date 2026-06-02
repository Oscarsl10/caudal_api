import os
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from prophet import Prophet

# Ruta de importación de la estructura modular
from app.schemas.forecast import ModelMetrics

VALID_MODELS = ["Ensamble", "Holt-Winters", "SARIMA", "Prophet"]

# Directorio persistente unificado con los artefactos de train_models.py
MODEL_DIR = "/app/models_store"
os.makedirs(MODEL_DIR, exist_ok=True)


def create_lag_features(series: pd.Series, lags=[1, 7, 14, 30]) -> pd.DataFrame:
    """
    Genera las características base replicando fielmente las transformaciones temporales
    y lags requeridos por el modelo de Gradient Boosting con desfase de shift(1).
    """
    df_features = pd.DataFrame(index=series.index)
    df_features["target"] = series

    for lag in lags:
        df_features[f"lag_{lag}"] = series.shift(lag)

    for window in [7, 14]:
        df_features[f"rolling_mean_{window}"] = series.shift(1).rolling(window=window).mean()
        df_features[f"rolling_std_{window}"] = series.shift(1).rolling(window=window).std()

    # Características temporales deterministas
    df_features["day_of_year"] = series.index.dayofyear
    df_features["month"] = series.index.month
    df_features["day_of_week"] = series.index.dayofweek

    return df_features


def calculate_metrics_frozen() -> dict:
    """
    Retorna el diccionario estático de validación cruzada científica obtenido en el Notebook.
    """
    return {
        "Ensamble": ModelMetrics(rmse=0.0749, mae=0.0552, mape=2.13, nse=0.9991, r2=0.9991, mda=87.1),
        "Holt-Winters": ModelMetrics(rmse=0.3129, mae=0.2451, mape=9.71, nse=-1.1148, r2=-1.1148, mda=57.63),
        "SARIMA": ModelMetrics(rmse=0.4608, mae=0.3512, mape=9.32, nse=-3.4466, r2=-3.4466, mda=57.63),
        "Prophet": ModelMetrics(rmse=2.1908, mae=1.9234, mape=47.83, nse=-102.6846, r2=-102.6846, mda=47.46)
    }


def generate_forecast_pipeline(series: pd.Series, horizon: int, best_model: str) -> pd.Series:
    """
    Orquesta los algoritmos de pronóstico de caudales y gestiona los artefactos guardados.
    """
    series = series.sort_index()

    last_date = series.index[-1]
    forecast_index = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=horizon, freq="D")

    if best_model == "SARIMA":
        model = SARIMAX(series, order=(1, 1, 1), seasonal_order=(1, 0, 0, 30), enforce_stationarity=False)
        res = model.fit(disp=False)
        return pd.Series(res.forecast(steps=horizon).values, index=forecast_index)

    elif best_model == "Holt-Winters":
        model = ExponentialSmoothing(series, trend="add", seasonal="add", seasonal_periods=30)
        res = model.fit()
        return pd.Series(res.forecast(steps=horizon).values, index=forecast_index)

    elif best_model == "Prophet":
        df_p = series.reset_index()
        df_p.columns = ["ds", "y"]
        model = Prophet(yearly_seasonality=True, daily_seasonality=False, weekly_seasonality=False)
        model.fit(df_p)
        future = model.make_future_dataframe(periods=horizon, freq="D")
        forecast = model.predict(future)
        return pd.Series(forecast["yhat"].iloc[-horizon:].values, index=forecast_index)

    else:
        # ==========================================================
        # MODELO ENSAMBLE (CORREGIDO USANDO PICKLE Y DESFASE CORRECTO)
        # ==========================================================
        pkl_path = os.path.join(MODEL_DIR, "ensemble_model.pkl")
        cols_path = os.path.join(MODEL_DIR, "ensemble_feature_cols.pkl")
        scaler_path = os.path.join(MODEL_DIR, "scaler_features.pkl")

        if os.path.exists(pkl_path):
            with open(pkl_path, "rb") as fh:
                gbm = pickle.load(fh)
            with open(cols_path, "rb") as fh:
                feature_cols = pickle.load(fh)
            with open(scaler_path, "rb") as fh:
                scaler = pickle.load(fh)
        else:
            # Fallback adaptativo
            df_features = create_lag_features(series).dropna()
            X_train = df_features.drop("target", axis=1)
            y_train = df_features["target"]
            feature_cols = X_train.columns.tolist()
            scaler = None

            gbm = GradientBoostingRegressor(
                n_estimators=200, max_depth=5, learning_rate=0.05, subsample=0.8, random_state=42
            )
            gbm.fit(X_train, y_train)

        # Clonación de la serie estructurada para simulación recursiva paso a paso
        history = series.copy()
        forecast_values = []

        for _ in range(horizon):
            next_date = history.index[-1] + pd.Timedelta(days=1)
            
            # 🔥 CORRECCIÓN MATEMÁTICA CLAVE: Las ventanas móviles deben calcularse 
            # excluyendo el paso actual para alinearse con el shift(1) del entrenamiento.
            row_dict = {
                "lag_1": history.iloc[-1],
                "lag_7": history.iloc[-7] if len(history) >= 7 else history.iloc[-1],
                "lag_14": history.iloc[-14] if len(history) >= 14 else history.iloc[-1],
                "lag_30": history.iloc[-30] if len(history) >= 30 else history.iloc[-1],
                "rolling_mean_7": history.iloc[-7:].mean(),
                "rolling_std_7": history.iloc[-7:].std() if len(history) >= 2 else 0.0,
                "rolling_mean_14": history.iloc[-14:].mean(),
                "rolling_std_14": history.iloc[-14:].std() if len(history) >= 2 else 0.0,
                "day_of_year": next_date.dayofyear,
                "month": next_date.month,
                "day_of_week": next_date.dayofweek
            }
            
            # Construir e indexar vector de entrada ordenadamente
            df_step = pd.DataFrame([row_dict])[feature_cols]
            
            # Transformación de escala idéntica
            if scaler is not None:
                X_step = scaler.transform(df_step)
            else:
                X_step = df_step.to_numpy()
            
            pred_val = float(gbm.predict(X_step)[0])
            pred_val = max(0.0, pred_val)  # Restricción física hidrológica
            
            forecast_values.append(pred_val)
            
            # Incorporar la predicción al historial para calcular los lags del siguiente día
            new_point = pd.Series([pred_val], index=[next_date])
            history = pd.concat([history, new_point])

        return pd.Series(forecast_values, index=forecast_index)