from fastapi import APIRouter, HTTPException, UploadFile, File, Form
import pandas as pd
from typing import Optional
import json

from app.schemas.forecast import ForecastRequest, ForecastResponse, HistoricRecord
from app.ml.engine import generate_forecast_pipeline, calculate_metrics_frozen, VALID_MODELS
from app.services.charts import generate_analytics_plots

router = APIRouter(prefix="/api/v1/forecast", tags=["Forecasting"])


def run_core_pipeline(df_input: pd.DataFrame, horizon: int, model_override: Optional[str]) -> dict:
    """
    Ejecuta el procesamiento común de datos, preparación de series, e inferencia ML.
    """
    df_input = df_input.sort_index()
    # Forzar frecuencia diaria implícita como exige statsmodels
    df_input = df_input.asfreq("D")
    
    if df_input["Caudal"].isna().sum() > 0:
        raise HTTPException(status_code=420, detail="La serie temporal contiene fechas faltantes o nulos tras alineación diaria.")

    serie_caudal = df_input["Caudal"]
    
    best_model = "Ensamble"
    if model_override and model_override in VALID_MODELS:
        best_model = model_override

    # Ejecutar motor predictivo
    forecast_series = generate_forecast_pipeline(serie_caudal, horizon, best_model)
    
    # Evaluar alertas tempranas del caudal medio proyectado
    f_mean = float(forecast_series.mean())
    alert_level = "LOW" if f_mean < 1.5 else "NORMAL" if f_mean < 8.0 else "HIGH"

    metrics_map = calculate_metrics_frozen()

    return {
        "best_model": best_model,
        "horizon_days": horizon,
        "forecast_dates": [str(d.date()) for d in forecast_series.index],
        "forecast_values": [round(float(v), 4) for v in forecast_series.values],
        "forecast_mean": round(f_mean, 4),
        "forecast_min": round(float(forecast_series.min()), 4),
        "forecast_max": round(float(forecast_series.max()), 4),
        "alert_level": alert_level,
        "metrics": metrics_map[best_model],
        "model_comparison": metrics_map,
        "plots": generate_analytics_plots(serie_caudal, forecast_series)
    }


@router.post("/manual", response_model=ForecastResponse)
async def forecast_manual_json(payload: ForecastRequest):
    """
    Recibe un arreglo estructurado vía JSON con validación integrada en Pydantic.
    """
    data_dict = {"Fecha": [r.fecha for r in payload.data], "Caudal": [r.caudal for r in payload.data]}
    df_input = pd.DataFrame(data_dict)
    df_input["Fecha"] = pd.to_datetime(df_input["Fecha"])
    df_input.set_index("Fecha", inplace=True)
    
    res = run_core_pipeline(df_input, payload.horizon_days, payload.model_override)
    return res


@router.post("/bulk", response_model=ForecastResponse)
async def forecast_bulk_excel(
    file: UploadFile = File(..., description="Archivo de Excel (.xlsx) conteniendo las columnas 'Fecha' y 'Caudal'"),
    horizon_days: int = Form(30, ge=1, le=90),
    model_override: Optional[str] = Form(None)
):
    """
    Carga masiva mediante análisis y validación estructural de archivos Excel (.xlsx).
    """
    if not file.filename.endswith((".xlsx", ".xls")):
        raise HTTPException(status_code=400, detail="Formato inválido de archivo. Debe ser extensión Excel (.xlsx).")

    try:
        df_excel = pd.read_excel(file.file)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"No se pudo parsear el archivo Excel debido a corrupción: {str(e)}")

    # Validaciones estructurales de Clean Code
    required_cols = {"Fecha", "Caudal"}
    if not required_cols.issubset(df_excel.columns):
        raise HTTPException(status_code=400, detail="El archivo Excel carece de las columnas mandatorias: 'Fecha' y 'Caudal'.")

    if len(df_excel) < 35:
        raise HTTPException(status_code=400, detail="Mínimo de registros insuficientes en Excel. Se requieren al menos 35 filas.")

    try:
        df_excel["Fecha"] = pd.to_datetime(df_excel["Fecha"])
        df_excel["Caudal"] = df_excel["Caudal"].astype(float)
    except Exception:
        raise HTTPException(status_code=422, detail="Error de tipado en datos: Fechas mal formateadas o Caudales no numéricos.")

    df_excel.set_index("Fecha", inplace=True)
    res = run_core_pipeline(df_excel, horizon_days, model_override)
    return res