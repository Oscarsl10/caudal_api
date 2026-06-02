import plotly.graph_objects as go
import pandas as pd


def generate_analytics_plots(historic: pd.Series, forecast: pd.Series) -> dict:
    """
    Genera componentes visuales para Datos Históricos, Predicciones y Comparativa unificada.
    """
    # 1. Gráfica Histórica (Últimos 120 días para contexto visual nítido)
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Scatter(x=historic.index[-120:], y=historic.values[-120:], mode="lines", name="Histórico (120d)", line=dict(color="#1f77b4")))
    fig_hist.update_layout(title="Monitoreo de Caudal Histórico Reciente", yaxis_title="Caudal (m³/s)", template="plotly_white")

    # 2. Gráfica del Pronóstico a Futuro
    fig_fore = go.Figure()
    fig_fore.add_trace(go.Scatter(x=forecast.index, y=forecast.values, mode="lines+markers", name="Pronóstico", line=dict(color="#FFA15A", width=2.5)))
    fig_fore.update_layout(title="Proyección de Tendencias de Caudal (Horizonte)", yaxis_title="Caudal (m³/s)", template="plotly_white")

    # 3. Gráfica Comparativa Unificada (Split visual entre pasado y futuro)
    fig_comp = go.Figure()
    fig_comp.add_trace(go.Scatter(x=historic.index[-90:], y=historic.values[-90:], mode="lines", name="Histórico Real", line=dict(color="#1f77b4")))
    fig_comp.add_trace(go.Scatter(x=forecast.index, y=forecast.values, mode="lines", name="Pronóstico del Modelo", line=dict(color="#FF6692", dash="dash")))
    
    # Línea vertical divisoria entre el pasado real y la predicción futurista
    fig_comp.add_shape(type="line", x0=historic.index[-1], x1=historic.index[-1], y0=0, y1=max(historic.max(), forecast.max()), line=dict(color="gray", width=1.5, dash="dot"))
    fig_comp.update_layout(title="Análisis Comparativo y Continuidad de la Serie", yaxis_title="Caudal (m³/s)", template="plotly_white")

    return {
        "plot_historic_html": fig_hist.to_html(include_plotlyjs="cdn", full_html=False),
        "plot_forecast_html": fig_fore.to_html(include_plotlyjs="cdn", full_html=False),
        "plot_comparison_html": fig_comp.to_html(include_plotlyjs="cdn", full_html=False)
    }