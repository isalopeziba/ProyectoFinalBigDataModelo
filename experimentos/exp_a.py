import json, sys
from functools import partial
from exp_comun import *

tr, va = cargar()
Xtr, ytr = tr[V.COLUMNAS_API], tr["es_fraude"].values
Xva, yva = va[V.COLUMNAS_API], va["es_fraude"].values

res = []
r, _ = evaluar("actual", lambda s: pipeline(construir_variables_v1, NUM_V1, seed=s), Xtr, ytr, Xva, yva)
res.append(r)

def con(extras):
    fn = partial(variables_v2, extras=tuple(extras))
    return lambda s: pipeline(fn, NUM_V1 + list(extras), seed=s)

r, _ = evaluar("A: todas", con(EXTRA_A), Xtr, ytr, Xva, yva); res.append(r)
for e in EXTRA_A:
    r, _ = evaluar(f"A: todas menos {e}", con([x for x in EXTRA_A if x != e]), Xtr, ytr, Xva, yva); res.append(r)
for e in EXTRA_A:
    r, _ = evaluar(f"A: actual + {e}", con([e]), Xtr, ytr, Xva, yva); res.append(r)

pd.DataFrame(res).to_csv(Path(__file__).parent / "res_a.csv", index=False)
