"""
esquemas.py
Contrato de entrada y salida de la API 
Si un campo no cumple estas reglas, la API responde 422 sin llamar al modelo.
"""
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator

import variables as V

ZONA_HORARIA = ZoneInfo("America/Bogota")


class Transaccion(BaseModel):
    """Lo que envía el backend por cada transacción."""
    model_config = ConfigDict(extra="ignore")  # si el backend manda campos de más, se ignoran

    transaccion_id: str = Field(min_length=1, max_length=40, examples=["TX-0083726"])
    monto: float = Field(gt=0, le=500_000_000, description="Monto en COP")
    fecha_hora: datetime = Field(description="Hora local de Colombia, ISO 8601")
    categoria: str = Field(description="Uno de los 14 nombres en español")
    monto_promedio_30d: float = Field(gt=0, description="Sin historial = monto actual")
    monto_promedio_categoria: float = Field(gt=0, description="Sin historial = monto actual")
    primera_vez_categoria: bool
    segundos_desde_ultima: float = Field(ge=0, description="Sin historial = 2592000")

    @field_validator("categoria")
    @classmethod
    def validar_categoria(cls, valor: str) -> str:
        nombre = V.normalizar_categoria(pd.Series([valor])).iloc[0]
        if nombre not in V.CATEGORIAS:
            raise ValueError(f"La categoría '{valor}' no existe. Valores válidos: {', '.join(V.CATEGORIAS)}")
        return nombre

    @field_validator("fecha_hora")
    @classmethod
    def a_hora_local(cls, valor: datetime) -> datetime:
        # El modelo aprendió con la hora local. Si llega con zona horaria (por ejemplo "Z" = UTC),
        # se convierte a hora de Colombia para no corromper la variable "hora".
        if valor.tzinfo is not None:
            valor = valor.astimezone(ZONA_HORARIA).replace(tzinfo=None)
        return valor

    def a_fila(self) -> dict:
        """Solo los campos que usa el modelo (V.COLUMNAS_API)."""
        datos = self.model_dump()
        return {columna: datos[columna] for columna in V.COLUMNAS_API}


class Prediccion(BaseModel):
    """Lo que la API devuelve al backend."""
    transaccion_id: str
    puntaje_riesgo: float = Field(ge=0, le=1)
    version_modelo: str


class LoteTransacciones(BaseModel):
    transacciones: list[Transaccion] = Field(min_length=1, max_length=10_000)


class LotePredicciones(BaseModel):
    predicciones: list[Prediccion]


class InfoModelo(BaseModel):
    version_modelo: str
    algoritmo: str
    fecha_entrenamiento: str
    umbral_sugerido: float
    criterio_umbral: str | None = None
    columnas_entrada: list[str]
    categorias: list[str]
    metricas_test: dict
