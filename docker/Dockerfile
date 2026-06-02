FROM python:3.11.10-slim

WORKDIR /app

# Dependencias del sistema:
#   libpq-dev + gcc  → psycopg2-binary
#   wget + tar       → descarga del binario CmdStan (requerido por Prophet/cmdstanpy)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev gcc wget tar \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Instalar el binario CmdStan que cmdstanpy/Prophet necesita en tiempo de ejecución.
# install_cmdstan compila Stan y lo deja en ~/.cmdstan; la versión debe coincidir
# con lo que cmdstanpy 1.2.4 espera (2.35.0).
RUN python -c "import cmdstanpy; cmdstanpy.install_cmdstan(version='2.35.0', overwrite=False)"

COPY . .

RUN chmod +x docker/entrypoint.sh

EXPOSE 8000

ENTRYPOINT ["docker/entrypoint.sh"]
