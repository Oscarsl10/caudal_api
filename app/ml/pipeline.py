"""
pipeline.py
-----------
Responsabilidad única: transformar la serie histórica en el vector de
lag-features que consume el modelo Ensamble (Gradient Boosting).

Las mismas fórmulas usadas en train_models.py deben replicarse aquí
para que el scaler ajustado en entrenamiento funcione correctamente
en producción.
"""

import numpy as np
import pandas as pd

# Lags utilizados en entrenamiento — no modificar sin reentrenar el modelo
LAGS = [1, 7, 14, 30]

# Orden fijo de features para el scaler (debe coincidir con FEATURE_COLS en train_models.py)
FEATURE_ORDER = (
    [f"lag_{l}" for l in LAGS]
    + ["rolling_mean_7", "rolling_mean_14", "rolling_std_7", "rolling_std_14"]
)


def compute_lag_features(serie: pd.Series) -> dict[str, float]:
    """
    Calcula las features de lag y rolling sobre los últimos valores de la serie.

    Parameters
    ----------
    serie : pd.Series
        Serie histórica de caudal diario con DatetimeIndex.
        Debe tener al menos max(LAGS) = 30 observaciones.

    Returns
    -------
    dict con las features calculadas en el mismo orden que FEATURE_ORDER.
    """
    if len(serie) < max(LAGS):
        raise ValueError(
            f"La serie debe tener al menos {max(LAGS)} observaciones; "
            f"se recibieron {len(serie)}."
        )

    features: dict[str, float] = {}

    for lag in LAGS:
        features[f"lag_{lag}"] = float(serie.iloc[-lag])

    features["rolling_mean_7"] = float(serie.tail(7).mean())
    features["rolling_mean_14"] = float(serie.tail(14).mean())
    features["rolling_std_7"] = float(serie.tail(7).std())
    features["rolling_std_14"] = float(serie.tail(14).std())

    return features


def normalize_features(features: dict[str, float], scaler) -> np.ndarray:
    """
    Transforma el diccionario de features en un vector normalizado [0,1]
    usando el scaler ajustado en entrenamiento.

    Parameters
    ----------
    features : dict
        Resultado de compute_lag_features().
    scaler : sklearn.preprocessing.MinMaxScaler
        Scaler cargado desde models_store/scaler_features.pkl.

    Returns
    -------
    np.ndarray de shape (1, n_features).
    """
    vector = np.array(
        [features[k] for k in FEATURE_ORDER], dtype=np.float64
    ).reshape(1, -1)
    return scaler.transform(vector)


def build_full_feature_matrix(serie: pd.Series) -> pd.DataFrame:
    """
    Construye la matriz de features para toda la serie histórica.
    Usado en train_models.py para ajustar el scaler.

    Parameters
    ----------
    serie : pd.Series
        Serie completa de caudal diario.

    Returns
    -------
    pd.DataFrame con columnas en el orden de FEATURE_ORDER y sin NaN
    (las primeras max(LAGS) filas se descartan).
    """
    df = pd.DataFrame(index=serie.index)
    for lag in LAGS:
        df[f"lag_{lag}"] = serie.shift(lag)
    df["rolling_mean_7"] = serie.shift(1).rolling(7).mean()
    df["rolling_mean_14"] = serie.shift(1).rolling(14).mean()
    df["rolling_std_7"] = serie.shift(1).rolling(7).std()
    df["rolling_std_14"] = serie.shift(1).rolling(14).std()
    df["target"] = serie.values
    return df.dropna()
