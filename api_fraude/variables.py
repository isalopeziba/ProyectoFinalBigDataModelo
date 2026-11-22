"""
variables.py
Definiciones compartidas de variables p

Este archivo lo usan:
  - el notebook de datos  para traducir categorias y calcular el historial,
  - el notebook del modelo (02_modelo.ipynb) dentro del pipeline guardado en .joblib,
  - la API de FastAPI, que debe importar este mismo modulo para poder cargar el .joblib.


"""

import unicodedata

import numpy as np
import pandas as pd

# Valor por defecto cuando el cliente no tiene transaccion anterior (debe coincidir con el backend)
SEGUNDOS_SIN_HISTORIAL = 30 * 24 * 3600  # 2.592.000 s = 30 dias


# ---------------------------------------------------------------------------
TRADUCCION_CATEGORIAS = {
    "entertainment": "entretenimiento",
    "food_dining": "alimentación y restaurantes",
    "gas_transport": "gasolina y transporte",
    "grocery_net": "compras de supermercado en línea",
    "grocery_pos": "compras en supermercado",
    "health_fitness": "salud y bienestar",
    "home": "hogar",
    "kids_pets": "niños y mascotas",
    "misc_net": "compras diversas en línea",
    "misc_pos": "compras diversas en establecimientos",
    "personal_care": "cuidado personal",
    "shopping_net": "compras en línea",
    "shopping_pos": "compras en establecimientos",
    "travel": "viajes",
}

CANAL_POR_CATEGORIA = {
    "compras de supermercado en línea": "en_linea",
    "compras diversas en línea": "en_linea",
    "compras en línea": "en_linea",
    "compras en supermercado": "punto_venta",
    "compras diversas en establecimientos": "punto_venta",
    "compras en establecimientos": "punto_venta",
    # las demas categorias -> "otro"
}

CATEGORIAS = sorted(TRADUCCION_CATEGORIAS.values())


def _sin_tildes(texto: str) -> str:
    texto = unicodedata.normalize("NFKD", str(texto).strip().lower())
    return "".join(c for c in texto if not unicodedata.combining(c))


# Acepta el codigo en ingles o el nombre en espanol, con o sin tildes y en cualquier mayuscula
_BUSQUEDA_CATEGORIA = {}
for _codigo, _nombre in TRADUCCION_CATEGORIAS.items():
    _BUSQUEDA_CATEGORIA[_sin_tildes(_codigo)] = _nombre
    _BUSQUEDA_CATEGORIA[_sin_tildes(_nombre)] = _nombre


def normalizar_categoria(categoria: pd.Series) -> pd.Series:
    """Devuelve el nombre oficial en espanol. Si no se reconoce, deja el valor original."""
    s = pd.Series(categoria).astype(str)
    return s.map(lambda c: _BUSQUEDA_CATEGORIA.get(_sin_tildes(c), c))


def canal_desde_categoria(categoria: pd.Series) -> pd.Series:
    """en_linea, punto_venta u otro, segun la tabla CANAL_POR_CATEGORIA."""
    return normalizar_categoria(categoria).map(CANAL_POR_CATEGORIA).fillna("otro")


# ---------------------------------------------------------------------------
# Contrato con la API
# ---------------------------------------------------------------------------
# Campos que el backend envia a POST /predict (ademas de transaccion_id).
# La distancia se elimino: en el EDA salio igual para fraude y normales
# asi que ya no se necesitan las coordenadas del cliente ni del comercio.
COLUMNAS_API = [
    "monto",
    "fecha_hora",
    "categoria",
    "monto_promedio_30d",
    "monto_promedio_categoria",
    "primera_vez_categoria",
    "segundos_desde_ultima",
]

# Variables que entran al modelo despues de construir_variables()
VARIABLES_NUMERICAS = [
    "log_monto",
    "hora",
    "dia_semana",
    "ratio_monto_30d",
    "ratio_monto_categoria",
    "primera_vez_categoria",
    "log_segundos_desde_ultima",
    
    "log_monto_promedio_30d",
    "log_monto_promedio_categoria",
    "ratio_categoria_vs_30d",
    "hora_decimal",
]
VARIABLES_CATEGORICAS = ["categoria", "canal"]


def calcular_historial(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula las 4 variables de historial usando SOLO transacciones anteriores del mismo cliente.
    Requiere columnas: cliente_id, fecha_hora (datetime), categoria, monto.
    Devuelve una copia ordenada por cliente y fecha con las columnas nuevas.
    """
    d = df.sort_values(["cliente_id", "fecha_hora"], kind="mergesort").copy()

    # Promedio de montos del cliente en los 30 dias previos (closed="left" excluye la actual)
    def _prom_30d(grupo: pd.DataFrame) -> pd.Series:
        s = grupo.set_index("fecha_hora")["monto"]
        return pd.Series(s.rolling("30D", closed="left").mean().values, index=grupo.index)

    d["monto_promedio_30d"] = (
        d.groupby("cliente_id", group_keys=False)[["fecha_hora", "monto"]].apply(_prom_30d)
    )

    # Promedio de montos anteriores del cliente en esa categoria (shift excluye la actual)
    d["monto_promedio_categoria"] = d.groupby(["cliente_id", "categoria"])["monto"].transform(
        lambda s: s.shift().expanding().mean()
    )

    # 1 si el cliente nunca habia comprado en esa categoria
    d["primera_vez_categoria"] = (
        d.groupby(["cliente_id", "categoria"]).cumcount().eq(0).astype(int)
    )

    # Segundos desde la transaccion anterior del cliente
    d["segundos_desde_ultima"] = d.groupby("cliente_id")["fecha_hora"].diff().dt.total_seconds()

    # Valores por defecto sin historial
    d["monto_promedio_30d"] = d["monto_promedio_30d"].fillna(d["monto"])
    d["monto_promedio_categoria"] = d["monto_promedio_categoria"].fillna(d["monto"])
    d["segundos_desde_ultima"] = d["segundos_desde_ultima"].fillna(SEGUNDOS_SIN_HISTORIAL)
    return d


def construir_variables(X: pd.DataFrame) -> pd.DataFrame:
    """
    Convierte los campos que llegan a la API (COLUMNAS_API) en las variables del modelo.
    Se usa dentro del pipeline (FunctionTransformer), asi el notebook y la API calculan igual.
    """
    X = pd.DataFrame(X).copy()
    fecha = pd.to_datetime(X["fecha_hora"])
    monto = X["monto"].astype(float)
    categoria = normalizar_categoria(X["categoria"])

    prom_30d = X["monto_promedio_30d"].astype(float).clip(lower=1.0)
    prom_cat = X["monto_promedio_categoria"].astype(float).clip(lower=1.0)

    out = pd.DataFrame(index=X.index)
    out["log_monto"] = np.log1p(monto)
    out["hora"] = fecha.dt.hour
    out["dia_semana"] = fecha.dt.day_of_week
    out["ratio_monto_30d"] = monto / prom_30d
    out["ratio_monto_categoria"] = monto / prom_cat
    out["primera_vez_categoria"] = X["primera_vez_categoria"].astype(int)
    out["log_segundos_desde_ultima"] = np.log1p(X["segundos_desde_ultima"].astype(float).clip(lower=0))
    # Nivel de gasto habitual del cliente (en general y en esa categoria)
    out["log_monto_promedio_30d"] = np.log1p(prom_30d)
    out["log_monto_promedio_categoria"] = np.log1p(prom_cat)
    # Que tan cara es esa categoria para el cliente comparada con su gasto general
    out["ratio_categoria_vs_30d"] = prom_cat / prom_30d
    # Hora con minutos (el pico de fraude empieza a las 22:00)
    out["hora_decimal"] = fecha.dt.hour + fecha.dt.minute / 60.0
    out["categoria"] = categoria.values
    out["canal"] = canal_desde_categoria(categoria).values
    return out[VARIABLES_NUMERICAS + VARIABLES_CATEGORICAS]
