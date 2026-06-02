from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import forecast

app = FastAPI(
    title="API de Pronóstico Hidrológico Integrado (IDEAM)",
    version="2.0.0",
    description="Microservicio de alta precisión para el pronóstico y alerta de caudales diarios utilizando arquitecturas de ensamble Gradient Boosting y modelos clásicos."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
    
app.include_router(forecast.router)


@app.get("/health", tags=["Infraestructura"])
async def health_check():
    return {"status": "healthy", "pipeline_sync": True}