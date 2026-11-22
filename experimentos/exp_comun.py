import sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

BASE = Path("C:/Users/isalo/Desktop/universidad/9.Semestre_9/Big data/Proyecto Final")
sys.path.insert(0, str(BASE))
import variables as V  # noqa: E402

RP = BASE / "datos_procesados"


def cargar(con_test=False):
    tr = pd.read_parquet(RP / "train.parquet")
    va = pd.read_parquet(RP / "valid.parquet")
    if con_test:
        return tr, va, pd.read_parquet(RP / "test.parquet")
    return tr, va


# ---------------- modelo 1.0.0 congelado ----------------
# variables.py ya incluye las variables del bloque A (v1.1.0). Para que estos experimentos sigan
# comparando contra el modelo 1.0.0, aqui se fijan las 7 variables numericas originales.
NUM_V1 = ["log_monto", "hora", "dia_semana", "ratio_monto_30d", "ratio_monto_categoria",
          "primera_vez_categoria", "log_segundos_desde_ultima"]


def construir_variables_v1(X):
    return V.construir_variables(X)[NUM_V1 + V.VARIABLES_CATEGORICAS]


# ---------------- variables (bloque A) ----------------
EXTRA_A = ["log_monto_promedio_30d", "log_monto_promedio_categoria", "diferencia_monto_30d",
           "ratio_categoria_vs_30d", "sin_historial", "hora_decimal"]


def variables_v2(X, extras=tuple(EXTRA_A)):
    X = pd.DataFrame(X)
    out = construir_variables_v1(X)
    monto = X["monto"].astype(float)
    p30 = X["monto_promedio_30d"].astype(float).clip(lower=1.0)
    pcat = X["monto_promedio_categoria"].astype(float).clip(lower=1.0)
    fecha = pd.to_datetime(X["fecha_hora"])
    todas = {
        "log_monto_promedio_30d": np.log1p(p30),
        "log_monto_promedio_categoria": np.log1p(pcat),
        "diferencia_monto_30d": monto - p30,
        "ratio_categoria_vs_30d": pcat / p30,
        "sin_historial": (X["segundos_desde_ultima"].astype(float) >= V.SEGUNDOS_SIN_HISTORIAL).astype(int),
        "hora_decimal": fecha.dt.hour + fecha.dt.minute / 60.0,
    }
    for e in extras:
        out[e] = np.asarray(todas[e])
    cat = out[V.VARIABLES_CATEGORICAS]
    return pd.concat([out.drop(columns=V.VARIABLES_CATEGORICAS), cat], axis=1)


def pipeline(fn_vars, numericas, lr=0.1, hojas=31, max_iter=300, class_weight="balanced",
             min_samples_leaf=20, l2=0.0, nativas=False, seed=42, extra_cols=None):
    if isinstance(class_weight, (int, float)):  # peso numérico de la clase fraude
        class_weight = {0: 1, 1: class_weight}
    if nativas:
        prep = ColumnTransformer([
            ("num", "passthrough", numericas),
            ("cat", "passthrough", V.VARIABLES_CATEGORICAS),
        ], verbose_feature_names_out=False).set_output(transform="pandas")
        cat_feat = "from_dtype"
        to_cat = FunctionTransformer(lambda d: d.astype({c: "category" for c in V.VARIABLES_CATEGORICAS}))
        pasos = [("variables", FunctionTransformer(fn_vars)), ("prep", prep), ("cat", to_cat)]
    else:
        prep = ColumnTransformer([
            ("num", "passthrough", numericas),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), V.VARIABLES_CATEGORICAS),
        ])
        cat_feat = None
        pasos = [("variables", FunctionTransformer(fn_vars)), ("prep", prep)]
    modelo = HistGradientBoostingClassifier(
        learning_rate=lr, max_leaf_nodes=hojas, max_iter=max_iter, class_weight=class_weight,
        min_samples_leaf=min_samples_leaf, l2_regularization=l2, early_stopping=True,
        random_state=seed, categorical_features=cat_feat)
    return Pipeline(pasos + [("modelo", modelo)])


# ---------------- métricas ----------------
def precision_at_k(y, p, k):
    idx = np.argsort(-p, kind="stable")[:min(k, len(y))]
    return y[idx].mean()


def fp_a_recall(y, p, recall_obj):
    """Falsos positivos con el umbral más alto que alcanza recall_obj."""
    prec, rec, thr = precision_recall_curve(y, p)
    ok = np.where(rec[:-1] >= recall_obj)[0]
    i = ok[-1]  # umbral más alto con ese recall
    tp = rec[i] * y.sum()
    return int(round(tp / prec[i] - tp))


def metricas_rank(y, p):
    return {"pr_auc": average_precision_score(y, p),
            "p@100": precision_at_k(y, p, 100),
            "p@500": precision_at_k(y, p, 500),
            "p@1000": precision_at_k(y, p, 1000),
            "fp@rec90": fp_a_recall(y, p, 0.90),
            "fp@rec95": fp_a_recall(y, p, 0.95)}


def evaluar(nombre, crear, Xtr, ytr, Xva, yva, seeds=(42, 7, 123)):
    filas, preds = [], []
    for s in seeds:
        t = time.time()
        m = crear(s).fit(Xtr, ytr)
        p = m.predict_proba(Xva)[:, 1]
        preds.append(p)
        r = metricas_rank(yva, p)
        r.update(seed=s, n_iter=m[-1].n_iter_, seg=round(time.time() - t))
        filas.append(r)
    df = pd.DataFrame(filas)
    res = df.drop(columns="seed").mean().to_dict()
    res["pr_auc_std"] = df["pr_auc"].std()
    res["exp"] = nombre
    print(f"{nombre:45s} PR-AUC={res['pr_auc']:.4f}±{res['pr_auc_std']:.4f} "
          f"p@500={res['p@500']:.3f} p@1000={res['p@1000']:.3f} fp@r90={res['fp@rec90']:.0f} "
          f"fp@r95={res['fp@rec95']:.0f} iter={res['n_iter']:.0f} t={res['seg']:.0f}s", flush=True)
    return res, np.mean(preds, axis=0)


# ---------------- umbral (bloque C) ----------------
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


