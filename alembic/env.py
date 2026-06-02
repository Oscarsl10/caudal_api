import os
from logging.config import fileConfig
from sqlalchemy import create_engine, pool
from alembic import context

config = context.config
if config.config_file_name:
    fileConfig(config.config_file_name)

# DATABASE_URL desde variable de entorno tiene prioridad absoluta sobre alembic.ini
database_url = os.environ.get("DATABASE_URL")
if not database_url:
    raise RuntimeError(
        "La variable de entorno DATABASE_URL no está definida. "
        "Verifica que el archivo .env existe y que docker-compose lo carga correctamente."
    )

# Sobreescribir siempre la URL del ini con la del entorno
config.set_main_option("sqlalchemy.url", database_url)

from app.database import Base      # noqa: E402
import app.models.forecast         # noqa: E402, F401

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    context.configure(
        url=database_url,
        target_metadata=target_metadata,
        literal_binds=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = create_engine(database_url, poolclass=pool.NullPool)
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
