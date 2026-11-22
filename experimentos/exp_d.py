"""Bloque D: SOLO medición. Variables que el backend tendría que calcular (no se agregan al modelo)."""
from functools import partial
from exp_comun import *

tr, va = cargar()
df = pd.concat([tr, va], ignore_index=True)
df["fecha_hora"] = pd.to_datetime(df["fecha_hora"])
df = df.sort_values(["cliente_id", "fecha_hora"], kind="mergesort")


def ventanas(g):
    s = g.set_index("fecha_hora")["monto"]
    out = pd.DataFrame(index=g.index)
    out["tx_1h"] = s.rolling("1h", closed="left").count().fillna(0).values
    out["tx_24h"] = s.rolling("24h", closed="left").count().fillna(0).values
    out["monto_24h"] = s.rolling("24h", closed="left").sum().fillna(0).values
    out["std_30d"] = s.rolling("30D", closed="left").std().values
    return out

w = df.groupby("cliente_id", group_keys=False)[["fecha_hora", "monto"]].apply(ventanas)
df = df.join(w)
df["tx_previas"] = df.groupby("cliente_id").cumcount()
df["zscore_30d"] = ((df["monto"] - df["monto_promedio_30d"]) / df["std_30d"].clip(lower=1.0)).fillna(0)
df = df.sort_index()

EXTRA_D = ["tx_1h", "tx_24h", "monto_24h", "tx_previas", "zscore_30d"]
cols = V.COLUMNAS_API + EXTRA_D
es_tr = df["particion"].eq("train").values
Xtr, ytr = df.loc[es_tr, cols], df.loc[es_tr, "es_fraude"].values
Xva, yva = df.loc[~es_tr, cols], df.loc[~es_tr, "es_fraude"].values

print(df.groupby("es_fraude")[EXTRA_D].median().T, flush=True)

import sys
mejor_a = [e for e in sys.argv[1].split(",") if e] if len(sys.argv) > 1 else EXTRA_A


def fn_vars(X, extras_a, extras_d):
    out = variables_v2(X, extras=tuple(extras_a))
    for e in extras_d:
        out[e] = X[e].values
    return out


def crear(extras_a, extras_d):
    fn = partial(fn_vars, extras_a=tuple(extras_a), extras_d=tuple(extras_d))
    return lambda s: pipeline(fn, NUM_V1 + list(extras_a) + list(extras_d), seed=s)

res = []
for nombre, ea, ed in [("D: actual + todas D", [], EXTRA_D),
                       ("D: A elegidas + todas D", mejor_a, EXTRA_D)] + \
                      [(f"D: A elegidas + {e}", mejor_a, [e]) for e in EXTRA_D]:
    r, _ = evaluar(nombre, crear(ea, ed), Xtr, ytr, Xva, yva)
    res.append(r)
pd.DataFrame(res).to_csv(Path(__file__).parent / "res_d.csv", index=False)
