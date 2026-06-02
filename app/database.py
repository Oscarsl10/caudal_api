from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, DeclarativeBase, Session
from app.config import settings


class Base(DeclarativeBase):
    pass


def _make_engine():
    """Crea el engine en tiempo de llamada, no en tiempo de importación.
    Esto garantiza que DATABASE_URL ya esté disponible como variable de entorno
    cuando Alembic o FastAPI importen este módulo.
    """
    return create_engine(settings.database_url)


def get_db():
    engine = _make_engine()
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db: Session = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def check_db_connection() -> bool:
    try:
        engine = _make_engine()
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False
