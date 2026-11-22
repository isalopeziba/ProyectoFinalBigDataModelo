"""
main.py
Servicio de IA del Proyecto 12: recibe transacciones del backend y devuelve el puntaje de riesgo.

La API NO decide si una transacción es "Sospechosa": solo devuelve el puntaje.
El backend compara ese puntaje con el umbral y aplica las reglas de negocio.

Ejecutar en local:   uvicorn main:app --reload --port 8000
Documentación:       http://localhost:8000/docs
"""
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

import variables as V  # necesario para que joblib pueda reconstruir el pipeline
from esquemas import InfoModelo, LotePredicciones, LoteTransacciones, Prediccion, Transaccion

RUTA_MODELOS = Path(os.getenv("RUTA_MODELOS", Path(__file__).parent / "modelos"))
log = logging.getLogger("servicio_ia")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

estado: dict = {}  # aquí quedan el modelo y su metadata, cargados una sola vez


@asynccontextmanager
async def ciclo_de_vida(app: FastAPI):
    # Se carga el modelo al iniciar (no en cada petición) para responder rápido
    estado["modelo"] = joblib.load(RUTA_MODELOS / "fraud_model.joblib")
    estado["metadata"] = json.loads((RUTA_MODELOS / "metadata_modelo.json").read_text(encoding="utf-8"))
    log.info("Modelo %s cargado desde %s", estado["metadata"]["version_modelo"], RUTA_MODELOS)
    yield
    estado.clear()


app = FastAPI(
    title="Servicio de IA – Detección de transacciones sospechosas",
    version="1.1.0",
    description="Recibe transacciones del backend y devuelve un puntaje de riesgo entre 0 y 1.",
    lifespan=ciclo_de_vida,
)


# ---------------- Errores con formato del contrato ----------------
@app.exception_handler(RequestValidationError)
async def datos_invalidos(request: Request, exc: RequestValidationError):
    detalle = [
        {"campo": ".".join(str(p) for p in e["loc"][1:]), "mensaje": e["msg"].removeprefix("Value error, ")}
        for e in exc.errors()
    ]
    return JSONResponse(status_code=422, content={"error": "datos_invalidos", "detalle": detalle})


@app.exception_handler(Exception)
async def error_inesperado(request: Request, exc: Exception):
    log.exception("Error inesperado")
    return JSONResponse(status_code=500, content={"error": "error_modelo", "detalle": str(exc)})


# ---------------- Funciones internas ----------------
def puntajes(transacciones: list[Transaccion]) -> list[float]:
    entrada = pd.DataFrame([t.a_fila() for t in transacciones], columns=V.COLUMNAS_API)
    return estado["modelo"].predict_proba(entrada)[:, 1].tolist()


def version() -> str:
    return estado["metadata"]["version_modelo"]


# ---------------- Endpoints ----------------
@app.get("/health", summary="¿El servicio está vivo y el modelo cargado?")
def health():
    return {"estado": "ok", "modelo_cargado": "modelo" in estado, "version_modelo": version()}


@app.get("/model-info", response_model=InfoModelo, summary="Versión, umbral sugerido y métricas del modelo")
def model_info():
    m = estado["metadata"]
    return InfoModelo(
        version_modelo=m["version_modelo"], algoritmo=m["algoritmo"],
        fecha_entrenamiento=m["fecha_entrenamiento"], umbral_sugerido=m["umbral_sugerido"],
        criterio_umbral=m.get("criterio_umbral"), columnas_entrada=m["columnas_entrada"],
        categorias=V.CATEGORIAS, metricas_test=m["metricas_test"],
    )


@app.post("/predict", response_model=Prediccion, summary="Puntaje de riesgo de una transacción")
def predict(transaccion: Transaccion):
    inicio = time.perf_counter()
    puntaje = puntajes([transaccion])[0]
    log.info("predict %s puntaje=%.4f %.1f ms", transaccion.transaccion_id, puntaje,
             (time.perf_counter() - inicio) * 1000)
    return Prediccion(transaccion_id=transaccion.transaccion_id,
                      puntaje_riesgo=round(puntaje, 4), version_modelo=version())


@app.post("/predict/batch", response_model=LotePredicciones,
          summary="Puntajes de varias transacciones (por ejemplo, al importar un CSV)")
def predict_batch(lote: LoteTransacciones):
    inicio = time.perf_counter()
    valores = puntajes(lote.transacciones)
    log.info("predict/batch %d transacciones %.1f ms", len(valores), (time.perf_counter() - inicio) * 1000)
    return LotePredicciones(predicciones=[
        Prediccion(transaccion_id=t.transaccion_id, puntaje_riesgo=round(p, 4), version_modelo=version())
        for t, p in zip(lote.transacciones, valores)
    ])
