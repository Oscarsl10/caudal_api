# API de Pronóstico de Caudal — Estación Pueblo Nuevo (IDEAM)

API REST construida con FastAPI que expone los cuatro modelos de pronóstico
desarrollados en el Notebook 4 del proyecto:

| Modelo | Descripción |
|---|---|
| SARIMA(1,1,1)(1,0,0)₃₀ | Modelo paramétrico con estacionalidad mensual |
| Holt-Winters (aditivo) | Suavización exponencial triple, período=30 |
| Prophet | Modelo bayesiano con detección de cambios de tendencia |
| **Ensamble (Gradient Boosting)** ⭐ | Lag-features (t-1, t-7, t-14, t-30) — **mejor modelo** |

---

## Flujo de una predicción

```
POST /forecast  { horizon_days: 30 }
        │
        ▼
app/routers/forecast.py
        │
        ▼
app/services/forecast_service.py
        ├─► app/ml/ensemble.py       → pronóstico con el mejor modelo
        ├─► app/ml/classifier.py     → nivel de alerta hidrológica
        ├─► PostgreSQL (ORM Forecast)
        └─► ForecastResponse (Pydantic)
```

---

## Estructura del proyecto

```
caudal_api/
├── docker/
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── entrypoint.sh
├── alembic/
│   ├── env.py
│   ├── script.py.mako
│   └── versions/
│       └── 0001_initial.py
├── app/
│   ├── main.py                  # FastAPI, lifespan, CORS, router
│   ├── config.py                # variables de entorno (pydantic-settings)
│   ├── database.py              # engine, SessionLocal, Base, get_db
│   ├── models/
│   │   └── forecast.py          # ORM SQLAlchemy — tabla forecasts
│   ├── schemas/
│   │   └── forecast.py          # Pydantic — request y response
│   ├── ml/
│   │   ├── pipeline.py          # lag-features + normalización
│   │   ├── ensemble.py          # inferencia de los 4 modelos
│   │   └── classifier.py        # niveles de alerta hidrológica
│   ├── routers/
│   │   └── forecast.py          # POST /forecast  GET /health
│   └── services/
│       └── forecast_service.py  # orquesta pipeline → modelos → DB → respuesta
├── models_store/                # artefactos .pkl (generados por train_models.py)
├── data/                        # caudal_limpio_diario.csv (exportado en Semana 2)
├── train_models.py
├── requirements.txt
├── .env.example
├── alembic.ini
└── .flake8
```

---

## Setup local

```bash
# 1. Crear entorno
python -m venv .venv
source .venv/bin/activate       # Linux/Mac
.venv\Scripts\activate          # Windows

pip install -r requirements.txt

# 2. Variables de entorno
cp .env.example .env
# Editar .env con tus credenciales de PostgreSQL

# 3. Copiar el CSV limpio exportado en Semana 2
# cp /ruta/a/caudal_limpio_diario.csv data/

# 4. Entrenar los modelos y serializar artefactos
python train_models.py

# 5. Migración de base de datos
alembic upgrade head

# 6. Levantar la API
uvicorn app.main:app --reload
```

---

## Setup con Docker

```bash
# Desde la raíz del proyecto
docker compose --env-file .env -f docker/docker-compose.yml up --build
```

El `entrypoint.sh` ejecuta automáticamente `train_models.py` si los
artefactos no existen, aplica las migraciones y arranca el servidor.

---

## Endpoints

| Método | Ruta | Descripción |
|---|---|---|
| `POST` | `/forecast` | Genera pronóstico de caudal |
| `GET` | `/health` | Estado de la API y modelos cargados |

### Ejemplo de request

```json
POST /forecast
{
  "horizon_days": 30,
  "model_override": null
}
```

### Ejemplo de response

```json
{
  "forecast_id": "3fa85f64-...",
  "best_model": "Ensamble",
  "horizon_days": 30,
  "forecast_dates": ["2017-12-01", "..."],
  "forecast_values": [12.34, 13.21, "..."],
  "forecast_mean": 12.5,
  "forecast_min": 10.1,
  "forecast_max": 18.3,
  "alert_level": "LOW",
  "metrics": {
    "rmse": 0.0749, "mae": 0.0552, "mape": 2.13,
    "nse": 0.9991, "r2": 0.9991, "mda": 87.1
  },
  "model_comparison": {
    "SARIMA":       {"rmse": 0.4608, "mae": 0.3512, "mape": 9.32},
    "Holt-Winters": {"rmse": 0.3129, "mae": 0.2451, "mape": 9.71},
    "Prophet":      {"rmse": 2.1908, "mae": 1.9234, "mape": 47.83},
    "Ensamble":     {"rmse": 0.0749, "mae": 0.0552, "mape": 2.13}
  }
}
```

---

## Niveles de alerta hidrológica

| Nivel | Umbral (P90 del pronóstico) | Acción |
|---|---|---|
| `ESTIAJE` | < 5.8 m³/s | Déficit hídrico — plan de contingencia |
| `LOW` | 5.8 – 14.0 m³/s | Condición normal — rutina |
| `MODERATE` | 14.0 – 22.0 m³/s | Caudal elevado — verificar obras |
| `HIGH` | 22.0 – 28.0 m³/s | Monitoreo intensivo |
| `CRITICAL` | > 28.0 m³/s | Alerta temprana — notificar autoridades |

---

## Documentación interactiva

```
http://localhost:8000/docs      # Swagger UI
http://localhost:8000/redoc     # ReDoc
```

---

**Datos:** DHIME – IDEAM Colombia | Estación 21117100 – PUEBLO NUEVO  
**Serie:** Caudal Medio Diario (m³/s) | 2010–2017
