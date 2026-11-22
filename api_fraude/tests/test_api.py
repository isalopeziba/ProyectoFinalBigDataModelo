"""
Pruebas automatizadas de la API (PR-06, PR-15 y PR-20 del documento de modelado).
Ejecutar desde la carpeta api_fraude:   pytest -v
Requiere que modelos/ tenga fraud_model.joblib y metadata_modelo.json.
"""
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from main import app  # noqa: E402

VALIDA = {
    "transaccion_id": "TX-0083726",
    "monto": 245800,
    "fecha_hora": "2026-11-21T10:30:00",
    "categoria": "compras en línea",
    "monto_promedio_30d": 62400,
    "monto_promedio_categoria": 15200,
    "primera_vez_categoria": True,
    "segundos_desde_ultima": 540,
}


@pytest.fixture(scope="module")
def cliente():
    with TestClient(app) as c:  # "with" ejecuta la carga del modelo
        yield c


def test_health(cliente):
    r = cliente.get("/health")
    assert r.status_code == 200 and r.json()["modelo_cargado"] is True


def test_model_info(cliente):  # PR-15
    r = cliente.get("/model-info")
    assert r.status_code == 200
    datos = r.json()
    assert {"version_modelo", "umbral_sugerido", "metricas_test"} <= datos.keys()
    assert len(datos["categorias"]) == 14


def test_predict_valida(cliente):  # PR-06
    r = cliente.post("/predict", json=VALIDA)
    assert r.status_code == 200
    datos = r.json()
    assert datos["transaccion_id"] == "TX-0083726"
    assert 0 <= datos["puntaje_riesgo"] <= 1
    assert datos["version_modelo"]


def test_categoria_sin_tilde_o_en_ingles(cliente):
    for cat in ["compras en linea", "COMPRAS EN LÍNEA", "shopping_net"]:
        r = cliente.post("/predict", json={**VALIDA, "categoria": cat})
        assert r.status_code == 200, cat


def test_fecha_con_zona_horaria(cliente):
    # 15:30 UTC = 10:30 en Colombia: debe dar el mismo puntaje que la hora local
    local = cliente.post("/predict", json=VALIDA).json()["puntaje_riesgo"]
    utc = cliente.post("/predict", json={**VALIDA, "fecha_hora": "2026-11-21T15:30:00Z"}).json()["puntaje_riesgo"]
    assert local == utc


@pytest.mark.parametrize("cambio, campo", [  # PR-20: casos límite
    ({"monto": -5000}, "monto"),
    ({"monto": 0}, "monto"),
    ({"categoria": "zapatos"}, "categoria"),
    ({"fecha_hora": "ayer"}, "fecha_hora"),
    ({"segundos_desde_ultima": -1}, "segundos_desde_ultima"),
    ({"monto_promedio_30d": 0}, "monto_promedio_30d"),
])
def test_datos_invalidos(cliente, cambio, campo):
    r = cliente.post("/predict", json={**VALIDA, **cambio})
    assert r.status_code == 422
    assert r.json()["error"] == "datos_invalidos"
    assert r.json()["detalle"][0]["campo"] == campo


def test_campo_faltante(cliente):
    incompleta = {k: v for k, v in VALIDA.items() if k != "monto"}
    r = cliente.post("/predict", json=incompleta)
    assert r.status_code == 422 and r.json()["detalle"][0]["campo"] == "monto"


def test_servicio_sigue_vivo_tras_errores(cliente):
    cliente.post("/predict", json={**VALIDA, "monto": "mucho"})
    assert cliente.get("/health").status_code == 200


def test_batch(cliente):
    lote = {"transacciones": [VALIDA, {**VALIDA, "transaccion_id": "TX-2", "monto": 3_500_000}]}
    r = cliente.post("/predict/batch", json=lote)
    assert r.status_code == 200
    ids = [p["transaccion_id"] for p in r.json()["predicciones"]]
    assert ids == ["TX-0083726", "TX-2"]


def test_mismo_resultado_que_individual(cliente):
    individual = cliente.post("/predict", json=VALIDA).json()["puntaje_riesgo"]
    lote = cliente.post("/predict/batch", json={"transacciones": [VALIDA]}).json()
    assert lote["predicciones"][0]["puntaje_riesgo"] == individual
