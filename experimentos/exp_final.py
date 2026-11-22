"""Evaluación final ÚNICA en test. Todo (variables, hiperparámetros, criterio de umbral) se decidió en valid.
argv[1] = criterio de umbral elegido en el bloque C."""
import json, sys
from functools import partial
from exp_comun import *

criterio = sys.argv[1]
tr, va, te = cargar(con_test=True)
Xtr, ytr = tr[V.COLUMNAS_API], tr["es_fraude"].values
Xva, yva = va[V.COLUMNAS_API], va["es_fraude"].values
Xte, yte = te[V.COLUMNAS_API], te["es_fraude"].values

cfg = json.loads((Path(__file__).parent / "mejor_b.json").read_text())
modelos = {
    "Actual (umbral precisión≥0,50)": (lambda: pipeline(construir_variables_v1, NUM_V1, seed=42), "precision>=0.50"),
    f"Actual (umbral {criterio})": (lambda: pipeline(construir_variables_v1, NUM_V1, seed=42), criterio),
    f"Mejorado (umbral {criterio})": (lambda: pipeline(partial(variables_v2, extras=tuple(cfg["extras"])),
                                                       NUM_V1 + cfg["extras"], seed=42, **cfg["cfg"]), criterio),
    "Mejorado (umbral tasa_fp<=0.002, punto alternativo)": (None, "tasa_fp<=0.002"),
}
filas, cache = [], {}
for nombre, (crear, crit) in modelos.items():
    clave = nombre.split(" ")[0]
    if clave not in cache:
        m = crear().fit(Xtr, ytr)
        cache[clave] = (m.predict_proba(Xva)[:, 1], m.predict_proba(Xte)[:, 1])
    pva, pte = cache[clave]
    u = umbral(yva, pva, crit)  # umbral elegido SOLO con valid
    r = {"modelo": nombre, "umbral": u, **metricas_rank(yte, pte), **medir(yte, pte, u)}
    filas.append(r)
res = pd.DataFrame(filas).set_index("modelo")
pd.set_option("display.width", 250)
print(res.round(4).T.to_string())
res.to_csv(Path(__file__).parent / "res_final_test.csv")
