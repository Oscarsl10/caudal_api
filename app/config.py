from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str
    model_store_path: str = "./models_store"
    data_path: str = "./data"
    forecast_horizon: int = 30

    model_config = {
        "env_file": ".env",
        "extra": "ignore",
        "protected_namespaces": ("settings_",),  # evita el warning de Pydantic
    }


settings = Settings()
