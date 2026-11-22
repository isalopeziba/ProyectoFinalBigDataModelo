"""Bloque C: criterio de umbral. Se elige el umbral en la 1a mitad (temporal) de valid y se mide en la 2a mitad,
para ver qué criterio se sostiene cuando cambia la tasa de fraude. No usa test."""
import json, sys
from functools import partial
from exp_comun import *

tr, va = cargar()
va = va.assign(fecha_hora=pd.to_datetime(va["fecha_hora"])).sort_values("fecha_hora", kind="mergesort").reset_index(drop=True)
Xtr, ytr = tr[V.COLUMNAS_API], tr["es_fraude"].values
mitad = len(va) // 2
v1, v2 = va.iloc[:mitad], va.iloc[mitad:]
print(f"valid 1a mitad: tasa fraude {v1.es_fraude.mean():.4%} | 2a mitad: {v2.es_fraude.mean():.4%}")

cfg_mejor = json.loads((Path(__file__).parent / "mejor_b.json").read_text())
modelos = {
    "actual": lambda: pipeline(construir_variables_v1, NUM_V1, seed=42),
    "mejorado": lambda: pipeline(partial(variables_v2, extras=tuple(cfg_mejor["extras"])),
                                 NUM_V1 + cfg_mejor["extras"], seed=42, **cfg_mejor["cfg"]),
}


def umbral(y, p, criterio):
    prec, rec, thr = precision_recall_curve(y, p)
    prec, rec = prec[:-1], rec[:-1]
    neg = (y == 0).sum()
    if criterio == "precision>=0.50":
        ok = np.where(prec >= 0.50)[0]; return thr[ok[np.argmax(rec[ok])]]
    if criterio == "recall>=0.95":
        ok = np.where(rec >= 0.95)[0]; return thr[ok[-1]]
    if criterio.startswith("tasa_fp<="):
        objetivo = float(criterio.split("=")[1])
        s = np.sort(p[y == 0])[::-1]
        return s[int(objetivo * neg)]  # deja pasar como mucho esa fracción de normales
    if criterio.startswith("tasa_alertas="):
        objetivo = float(criterio.split("=")[1])
        return np.quantile(p, 1 - objetivo)


def medir(y, p, u):
    pred = p >= u
    tp = (pred & (y == 1)).sum(); fp = (pred & (y == 0)).sum()
    return {"precision": tp / max(tp + fp, 1), "recall": tp / y.sum(), "falsos_positivos": int(fp),
            "tasa_fp": fp / (y == 0).sum(), "tasa_alertas": pred.mean()}


filas = []
for nm, crear in modelos.items():
    m = crear().fit(Xtr, ytr)
    p1 = m.predict_proba(v1[V.COLUMNAS_API])[:, 1]
    p2 = m.predict_proba(v2[V.COLUMNAS_API])[:, 1]
    y1, y2 = v1.es_fraude.values, v2.es_fraude.values
    for c in ["precision>=0.50", "recall>=0.95", "tasa_fp<=0.004", "tasa_fp<=0.002", "tasa_alertas=0.009"]:
        u = umbral(y1, p1, c)
        r1, r2 = medir(y1, p1, u), medir(y2, p2, u)
        filas.append({"modelo": nm, "criterio": c, "umbral": u,
                      **{f"{k}_v1": v for k, v in r1.items()}, **{f"{k}_v2": v for k, v in r2.items()}})
res = pd.DataFrame(filas)
pd.set_option("display.width", 250)
print(res.round(4).to_string(index=False))
res.to_csv(Path(__file__).parent / "res_c.csv", index=False)
