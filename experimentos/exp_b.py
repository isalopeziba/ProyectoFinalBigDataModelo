"""Bloque B: hiperparámetros por etapas, todo en valid. argv[1] = extras de A separados por coma."""
import sys, json
from functools import partial
from exp_comun import *

extras = [e for e in sys.argv[1].split(",") if e] if len(sys.argv) > 1 else []
tr, va = cargar()
Xtr, ytr = tr[V.COLUMNAS_API], tr["es_fraude"].values
Xva, yva = va[V.COLUMNAS_API], va["es_fraude"].values
fn = partial(variables_v2, extras=tuple(extras))
num = NUM_V1 + extras

base = dict(lr=0.1, hojas=31, max_iter=300, class_weight="balanced", min_samples_leaf=20, l2=0.0, nativas=False)
res = []


def probar(nombre, **cambios):
    cfg = {**base, **cambios}
    r, _ = evaluar(nombre, lambda s: pipeline(fn, num, seed=s, **cfg), Xtr, ytr, Xva, yva)
    r["cfg"] = json.dumps(cfg)
    res.append(r)
    pd.DataFrame(res).to_csv(Path(__file__).parent / "res_b.csv", index=False)
    return r


def mejor_de(rs):
    """rs[0] es la configuracion vigente; un retador solo la reemplaza si mejora el PR-AUC en mas de 0.002."""
    actual, retador = rs[0], max(rs, key=lambda r: r["pr_auc"])
    return retador if retador["pr_auc"] > actual["pr_auc"] + 0.002 else actual

# Etapa 1: peso de clases
rs = [probar("B1 peso=balanced"),
      probar("B1 peso=None", class_weight=None),
      probar("B1 peso=1:10", class_weight=10),
      probar("B1 peso=1:30", class_weight=30)]
base.update(json.loads(mejor_de(rs)["cfg"])); print(">> etapa 1:", base, flush=True)

# Etapa 2: tasa de aprendizaje / iteraciones / hojas
rs = [r for r in res if json.loads(r["cfg"]) == base]
rs += [probar("B2 lr=0.05 iter=800", lr=0.05, max_iter=800),
       probar("B2 lr=0.1 iter=800", max_iter=800),
       probar("B2 hojas=15 iter=800", hojas=15, max_iter=800),
       probar("B2 hojas=63 iter=800", hojas=63, max_iter=800),
       probar("B2 hojas=127 lr=0.05 iter=800", hojas=127, lr=0.05, max_iter=800)]
base.update(json.loads(mejor_de(rs)["cfg"])); print(">> etapa 2:", base, flush=True)

# Etapa 3: regularización
rs = [r for r in res if json.loads(r["cfg"]) == base]
rs += [probar("B3 min_leaf=100", min_samples_leaf=100),
       probar("B3 min_leaf=300", min_samples_leaf=300),
       probar("B3 l2=1", l2=1.0),
       probar("B3 l2=10", l2=10.0)]
base.update(json.loads(mejor_de(rs)["cfg"])); print(">> etapa 3:", base, flush=True)

# Etapa 4: categorías nativas
rs = [r for r in res if json.loads(r["cfg"]) == base]
rs += [probar("B4 categorias nativas", nativas=True)]
base.update(json.loads(mejor_de(rs)["cfg"])); print(">> FINAL:", json.dumps(base), flush=True)
(Path(__file__).parent / "mejor_b.json").write_text(json.dumps({"extras": extras, "cfg": base}))
