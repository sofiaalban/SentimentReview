"""Matriz de confusión y métricas por clase, calculadas a mano (sin
sklearn.metrics) — es justo lo que pidió la profesora al final del enunciado.
"""
from typing import List, Sequence

import numpy as np
import pandas as pd


def matriz_confusion(y_true: Sequence[str], y_pred: Sequence[str], clases: List[str]) -> pd.DataFrame:
    idx = {c: i for i, c in enumerate(clases)}
    cm = np.zeros((len(clases), len(clases)), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[idx[t], idx[p]] += 1
    return pd.DataFrame(cm, index=[f"real:{c}" for c in clases], columns=[f"pred:{c}" for c in clases])


def metricas_por_clase(cm: pd.DataFrame, clases: List[str]) -> pd.DataFrame:
    cm_np = cm.to_numpy()
    filas = []
    for i, clase in enumerate(clases):
        vp = cm_np[i, i]
        fp = cm_np[:, i].sum() - vp
        fn = cm_np[i, :].sum() - vp
        precision = vp / (vp + fp) if (vp + fp) > 0 else 0.0
        recall = vp / (vp + fn) if (vp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        filas.append({"clase": clase, "precision": precision, "recall": recall, "f1": f1, "soporte": int(cm_np[i, :].sum())})
    return pd.DataFrame(filas).set_index("clase")


def accuracy(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float((y_true == y_pred).mean())


def graficar_matriz_confusion(cm: pd.DataFrame, titulo: str, ruta_salida: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm.to_numpy(), cmap="Blues")
    ax.set_xticks(range(len(cm.columns)))
    ax.set_xticklabels([c.replace("pred:", "") for c in cm.columns], rotation=45, ha="right")
    ax.set_yticks(range(len(cm.index)))
    ax.set_yticklabels([r.replace("real:", "") for r in cm.index])
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")
    ax.set_title(titulo)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            valor = cm.to_numpy()[i, j]
            color = "white" if valor > cm.to_numpy().max() / 2 else "black"
            ax.text(j, i, str(valor), ha="center", va="center", color=color)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(ruta_salida, dpi=150)
    plt.close(fig)
