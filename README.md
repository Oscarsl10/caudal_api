# API de Pronóstico Hidrológico Integrado (IDEAM) — Estación Pueblo Nuevo

API REST construida con FastAPI (versión 2.0.0) que pronostica el caudal medio diario
de 1 a 90 días a partir de una serie histórica enviada por el usuario, y asigna un nivel
de alerta a partir del promedio del pronóstico.

Expone cuatro modelos de pronóstico, desarrollados en el Notebook 4 del proyecto:

| Modelo (`model_override`) | Descripción |
|---|---|
| `Ensamble` ⭐ (predeterminado) | Gradient Boosting con variables de rezago (t-1, t-7, t-14, t-30), medias y desviaciones móviles de 7 y 14 días, y variables de calendario. Pronóstico recursivo día a día. Usa el modelo preentrenado de `models_store/`. |
| `SARIMA` | SARIMA(1,1,1)(1,0,0)₃₀. Se ajusta con los datos de cada solicitud. |
| `Holt-Winters` | Suavización exponencial triple aditiva, período 30. Se ajusta con los datos de cada solicitud. Requiere al menos 60 registros. |
| `Prophet` | Modelo con estacionalidad anual. Se ajusta con los datos de cada solicitud. |

---

## Flujo de una predicción

```
POST /api/v1/forecast/manual   (cuerpo JSON)
POST /api/v1/forecast/bulk     (archivo Excel .xlsx)
        │
        ▼
app/routers/forecast.py   → valida la serie (mínimo 35 registros, un registro por día)
        │
        ▼
app/ml/engine.py          → pronóstico con el modelo elegido (Ensamble por defecto)
        │
        ▼
nivel de alerta (LOW / NORMAL / HIGH) según el promedio del pronóstico
        │
        ▼
ForecastResponse (Pydantic) → respuesta JSON
```

La versión actual **no guarda** las solicitudes ni los resultados en base de datos.

---

## Estructura del proyecto

```
caudal_api/
├── docker/
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── entrypoint.sh
├── alembic/                     # migraciones de PostgreSQL (tabla forecasts)
├── app/
│   ├── main.py                  # FastAPI, CORS, router y GET /health
│   ├── config.py                # variables de entorno (pydantic-settings)
│   ├── schemas/
│   │   └── forecast.py          # Pydantic — request y response
│   ├── routers/
│   │   └── forecast.py          # POST /api/v1/forecast/manual y /bulk
│   ├── ml/
│   │   └── engine.py            # motor de pronóstico de los 4 modelos (en uso)
│   ├── services/
│   │   └── charts.py            # gráficos Plotly (generados, no incluidos en la respuesta)
│   │
│   │   # Módulos presentes pero no conectados a los endpoints actuales:
│   ├── database.py              # engine y sesión de SQLAlchemy (lo usa Alembic)
│   ├── models/forecast.py       # ORM de la tabla forecasts
│   ├── ml/pipeline.py           # variables de rezago (versión alterna)
│   ├── ml/ensemble.py           # inferencia de los 4 modelos (versión alterna)
│   ├── ml/classifier.py         # clasificación de cinco niveles (reservada para una versión futura)
│   └── services/forecast_service.py  # flujo con persistencia en base de datos
├── models_store/                # modelo Ensamble preentrenado (.pkl)
├── docs/                        # Manual Técnico (PDF), redoc.html, architecture.svg
├── train_models.py              # reentrenamiento (requiere data/caudal_limpio_diario.csv)
├── requirements.txt
├── .env                         # variables de entorno de ejemplo
├── alembic.ini
└── .flake8
```

---

## Requisitos

- **Docker (recomendado):** Docker Engine y Docker Compose.
- **Local:** Python 3.11 (la versión que usa el `Dockerfile`; la instalación de
  `requirements.txt` falla con Python 3.13).
- PostgreSQL solo se necesita para aplicar las migraciones; los endpoints actuales no
  leen ni escriben en la base de datos.

---

## Setup con Docker (recomendado)

```bash
# Desde la raíz del proyecto
docker compose --env-file .env -f docker/docker-compose.yml up --build
```

El `entrypoint.sh` entrena los modelos solo si no encuentra `ensemble_model.pkl` en
`models_store/` (el repositorio ya lo incluye), aplica las migraciones de Alembic y
arranca el servidor en el puerto 8000.

---

## Setup local

```bash
# 1. Crear entorno (Python 3.11)
python3.11 -m venv .venv
source .venv/bin/activate       # Linux/Mac
.venv\Scripts\activate          # Windows

pip install -r requirements.txt

# 2. Copiar el modelo preentrenado a la ruta que usa la API
sudo mkdir -p /app/models_store
sudo cp models_store/*.pkl /app/models_store/

# 3. Levantar la API (desde la raíz del proyecto)
uvicorn app.main:app --port 8000
```

La API busca los modelos siempre en `/app/models_store` (ruta fija en `app/ml/engine.py`;
en Windows equivale a `\app\models_store` en la unidad del proyecto). Sin ese paso la API
funciona, pero el `Ensamble` se entrena en el momento con los datos de cada solicitud.

El archivo `.env` incluido contiene valores de ejemplo (`DATABASE_URL`, `POSTGRES_USER`,
`POSTGRES_PASSWORD`, `POSTGRES_DB`, `MODEL_STORE_PATH`, `DATA_PATH`, `FORECAST_HORIZON`).
Cambie el usuario y la contraseña si publica el servicio en una red compartida.

Reentrenar los modelos (`python train_models.py`) solo es necesario si faltan los
archivos de `models_store/`; requiere el archivo `data/caudal_limpio_diario.csv`
(columnas `Fecha` y `Caudal`), que no se incluye en el repositorio.

---

## Endpoints

| Método | Ruta | Descripción |
|---|---|---|
| `GET` | `/health` | Comprueba que el servicio está activo (respuesta fija) |
| `POST` | `/api/v1/forecast/manual` | Pronóstico a partir de una serie enviada en JSON |
| `POST` | `/api/v1/forecast/bulk` | Pronóstico a partir de un archivo Excel (`.xlsx`) |

### Parámetros

| Campo | Tipo | Obligatorio | Reglas |
|---|---|---|---|
| `data` (solo `/manual`) | lista de `{fecha, caudal}` | Sí | Mínimo 35 registros (60 con Holt-Winters). `fecha` en `AAAA-MM-DD`, `caudal` ≥ 0 en m³/s. Un registro por día, sin huecos ni fechas repetidas. |
| `file` (solo `/bulk`) | archivo `.xlsx` | Sí | Primera hoja con columnas `Fecha` y `Caudal`; mínimo 35 filas. |
| `horizon_days` | entero | No (30) | De 1 a 90. |
| `model_override` | texto | No | `Ensamble`, `Holt-Winters`, `SARIMA` o `Prophet`. Un valor ausente o no reconocido usa `Ensamble`. |

### Ejemplo de request (`/manual`, extracto)

```json
{
  "horizon_days": 7,
  "model_override": null,
  "data": [
    { "fecha": "2017-11-02", "caudal": 2.976 },
    { "fecha": "2017-11-03", "caudal": 2.791 },
    { "fecha": "2017-11-04", "caudal": 2.812 }
  ]
}
```

El extracto solo muestra la forma del cuerpo: una solicitud real necesita al menos 35
registros diarios consecutivos.

### Ejemplo con Excel (`/bulk`)

```bash
curl -X POST "http://localhost:8000/api/v1/forecast/bulk" \
  -F "file=@datos_caudal.xlsx" -F "horizon_days=7" -F "model_override=SARIMA"
```

### Ejemplo de response

```json
{
  "forecast_id": "48ce436f-bafa-4d23-82bb-f49099df0091",
  "best_model": "Ensamble",
  "horizon_days": 7,
  "forecast_dates": ["2018-01-01", "2018-01-02", "2018-01-03", "2018-01-04", "2018-01-05", "2018-01-06", "2018-01-07"],
  "forecast_values": [2.4666, 2.5206, 2.647, 2.7605, 2.8818, 2.9303, 3.0126],
  "forecast_mean": 2.7456,
  "forecast_min": 2.4666,
  "forecast_max": 3.0126,
  "alert_level": "NORMAL",
  "metrics": { "rmse": 0.0749, "mae": 0.0552, "mape": 2.13, "nse": 0.9991, "r2": 0.9991, "mda": 87.1 },
  "model_comparison": {
    "Ensamble":     { "rmse": 0.0749, "mae": 0.0552, "mape": 2.13, "nse": 0.9991, "r2": 0.9991, "mda": 87.1 },
    "Holt-Winters": { "rmse": 0.3129, "mae": 0.2451, "mape": 9.71, "nse": -1.1148, "r2": -1.1148, "mda": 57.63 },
    "SARIMA":       { "rmse": 0.4608, "mae": 0.3512, "mape": 9.32, "nse": -3.4466, "r2": -3.4466, "mda": 57.63 },
    "Prophet":      { "rmse": 2.1908, "mae": 1.9234, "mape": 47.83, "nse": -102.6846, "r2": -102.6846, "mda": 47.46 }
  }
}
```

Las métricas son valores fijos de validación incorporados en la API: no se recalculan con
los datos enviados.

### Códigos de error

| Código | Causa |
|---|---|
| 400 | Excel: extensión inválida, faltan las columnas `Fecha` y `Caudal`, o menos de 35 filas. |
| 420 | La serie tiene días faltantes o valores vacíos (código no estándar usado por esta API). |
| 422 | Datos inválidos: menos de 35 registros, `horizon_days` fuera de 1–90, fechas o caudales inválidos, Excel ilegible. |
| 500 | Fechas repetidas en la serie, o Holt-Winters con menos de 60 registros. |

---

## Niveles de alerta hidrológica

El nivel se calcula con el **promedio** del pronóstico (`forecast_mean`):

| Nivel | Condición | Significado |
|---|---|---|
| `LOW` | promedio < 1.5 m³/s | Caudal bajo — seguimiento de rutina |
| `NORMAL` | 1.5 ≤ promedio < 8.0 m³/s | Condiciones normales y estables |
| `HIGH` | promedio ≥ 8.0 m³/s | Caudal elevado — monitoreo intensivo |

`app/ml/classifier.py` contiene una clasificación de cinco niveles (`ESTIAJE`, `LOW`,
`MODERATE`, `HIGH`, `CRITICAL`) basada en el percentil 90 del pronóstico, reservada para
una versión futura. La API actual no la utiliza.

---

## Consideraciones de uso

- El `Ensamble` preentrenado se calibró con la serie de la estación Pueblo Nuevo
  (2010–2017; caudal medio 3.48 m³/s y máximo 12.48 m³/s). Con caudales muy superiores a
  ese rango puede subestimar el pronóstico: compare con los demás modelos.
- `Holt-Winters` necesita al menos 60 registros (dos ciclos de 30 días).

---

## Documentación interactiva

```
http://localhost:8000/docs      # Swagger UI
http://localhost:8000/redoc     # ReDoc
http://localhost:8000/openapi.json
```

La carpeta `docs/` incluye el Manual Técnico y una página ReDoc estática (`redoc.html`).

---

**Datos:** DHIME – IDEAM Colombia | Estación 21117100 – PUEBLO NUEVO  
**Serie:** Caudal Medio Diario (m³/s) | 2010–2017
