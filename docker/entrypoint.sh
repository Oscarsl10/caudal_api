#!/bin/sh
set -e

# Si DATABASE_URL no llega como variable, construirla desde las partes individuales.
# Ambas formas son válidas y docker-compose garantiza que las variables de postgres sí llegan.
if [ -z "$DATABASE_URL" ]; then
    export DATABASE_URL="postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@db:5432/${POSTGRES_DB}"
    echo "INFO: DATABASE_URL construida desde variables individuales: postgresql://${POSTGRES_USER}:***@db:5432/${POSTGRES_DB}"
fi

if [ ! -f "$MODEL_STORE_PATH/ensemble_model.pkl" ]; then
    echo "Artefactos no encontrados — ejecutando train_models.py ..."
    python train_models.py
fi

echo "Aplicando migraciones Alembic ..."
alembic upgrade head

echo "Iniciando API de Pronostico de Caudal ..."
exec uvicorn app.main:app --host 0.0.0.0 --port 8000
