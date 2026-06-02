"""
classifier.py
-------------
Lógica de negocio hidrológica pura (sin ML):
- Clasifica el nivel de alerta según el caudal pronosticado.
- Genera las recomendaciones operacionales correspondientes.

Los umbrales están basados en percentiles históricos de la estación
Pueblo Nuevo (IDEAM 21117100). Ajustar si se incorporan otras estaciones.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional

# ─────────────────────────────────────────────────────────────────────────────
# Umbrales de alerta hidrológica (m³/s)
# Fuente: estadísticas descriptivas de la serie 2010-2017
# P25 ≈ 5.8  |  P50 ≈ 9.2  |  P75 ≈ 14.1  |  P90 ≈ 22.5  |  P95 ≈ 28.3
# ─────────────────────────────────────────────────────────────────────────────
ALERT_THRESHOLDS = {
    "CRITICAL": 28.0,   # > P95 — riesgo de avenida
    "HIGH": 22.0,       # > P90 — caudal elevado, monitoreo intensivo
    "MODERATE": 14.0,   # > P75 — caudal sobre la mediana alta
    "LOW": 5.8,         # > P25 — condición normal-baja
    # Por debajo de 5.8 → ESTIAJE
}

# Recomendaciones operacionales por nivel de alerta
RECOMMENDATIONS: dict[str, dict] = {
    "CRITICAL": {
        "actions": [
            "Activar protocolo de alerta temprana en la cuenca",
            "Notificar a autoridades locales y Defensa Civil",
            "Suspender captaciones directas del río",
            "Evaluar riesgo de desbordamiento en zonas bajas",
        ],
        "timeline": "Inmediato (< 24 h)",
        "owner": "Coordinador de Gestión del Riesgo",
        "frequency": "Monitoreo cada 6 horas",
        "metrics": ["Nivel del río", "Precipitación acumulada 24 h", "Caudal en tiempo real"],
    },
    "HIGH": {
        "actions": [
            "Incrementar frecuencia de aforo manual",
            "Alertar operadores de bocatomas aguas abajo",
            "Revisar capacidad de almacenamiento en embalses",
        ],
        "timeline": "24-48 horas",
        "owner": "Jefe de Operaciones Hídricas",
        "frequency": "Monitoreo cada 12 horas",
        "metrics": ["Caudal diario", "Turbidez", "Nivel en estación de aforo"],
    },
    "MODERATE": {
        "actions": [
            "Verificar estado de obras de protección",
            "Actualizar pronóstico con datos de precipitación más recientes",
        ],
        "timeline": "3-5 días",
        "owner": "Técnico de Recursos Hídricos",
        "frequency": "Revisión diaria",
        "metrics": ["Caudal diario", "Tendencia semanal"],
    },
    "LOW": {
        "actions": [
            "Condiciones normales — seguimiento de rutina",
            "Verificar datos del sensor cada 24 h",
        ],
        "timeline": "Rutinario",
        "owner": "Operador de Estación",
        "frequency": "Revisión diaria",
        "metrics": ["Caudal diario"],
    },
    "ESTIAJE": {
        "actions": [
            "Activar plan de contingencia por déficit hídrico",
            "Evaluar restricciones de captación",
            "Coordinar con usuarios aguas abajo",
        ],
        "timeline": "Inmediato",
        "owner": "Coordinador de Gestión del Riesgo",
        "frequency": "Monitoreo cada 12 horas",
        "metrics": ["Caudal mínimo diario", "Almacenamiento en embalses"],
    },
}


def classify_alert(forecast_values: list[float]) -> str:
    """
    Determina el nivel de alerta hidrológica a partir del pronóstico.

    Usa el percentil 90 de los valores pronosticados para evitar que
    un solo día extremo dispare una alerta CRITICAL innecesaria.

    Parameters
    ----------
    forecast_values : list[float]
        Valores de caudal pronosticados (m³/s).

    Returns
    -------
    str : "CRITICAL" | "HIGH" | "MODERATE" | "LOW" | "ESTIAJE"
    """
    representative_value = float(np.percentile(forecast_values, 90))

    if representative_value >= ALERT_THRESHOLDS["CRITICAL"]:
        return "CRITICAL"
    if representative_value >= ALERT_THRESHOLDS["HIGH"]:
        return "HIGH"
    if representative_value >= ALERT_THRESHOLDS["MODERATE"]:
        return "MODERATE"
    if representative_value >= ALERT_THRESHOLDS["LOW"]:
        return "LOW"
    return "ESTIAJE"


def get_recommendation(alert_level: str) -> Optional[dict]:
    """
    Recupera las recomendaciones operacionales para el nivel de alerta dado.

    Parameters
    ----------
    alert_level : str
        Resultado de classify_alert().

    Returns
    -------
    dict con acciones, responsable, frecuencia y métricas de seguimiento,
    o None si el nivel no está en la matriz.
    """
    return RECOMMENDATIONS.get(alert_level)
