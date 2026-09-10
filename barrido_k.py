"""Barrido de k para k-NN + Jaccard y k-NN + Coseno, reusando las matrices de
similitud ya calculadas (no hay que rehacer Jaccard/coseno por cada k, solo
cambia a cuántos vecinos se mira). Sirve para confirmar si un resultado a un
k fijo es consistente o es un accidente de esa elección puntual.
"""
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from src.texto import tokenize, split_train_test
from src.vectorize import Vectorizador
from src.similitud import jaccard_similitud, coseno_similitud
from src.clasificadores import KNNClasificador
from src.evaluacion import accuracy

df = pd.read_csv("data/muchocine_reviews.csv")
df_train, df_test = split_train_test(df, "sentimiento", test_size=0.2, seed=42)
train_tok = [tokenize(t) for t in df_train["review_body"]]
test_tok = [tokenize(t) for t in df_test["review_body"]]
y_train = df_train["sentimiento"].tolist()
y_test = df_test["sentimiento"].tolist()

vec = Vectorizador(min_df=2).fit(train_tok)
bin_train, bin_test = vec.transform_binaria(train_tok), vec.transform_binaria(test_tok)
tfidf_train, tfidf_test = vec.transform_tfidf(train_tok), vec.transform_tfidf(test_tok)

print("Calculando matrices de similitud (una sola vez)...")
t0 = time.perf_counter()
sim_jaccard = jaccard_similitud(bin_test, bin_train)
sim_coseno = coseno_similitud(tfidf_test, tfidf_train)
print(f"  {time.perf_counter()-t0:.2f}s\n")

ks = [1, 3, 5, 9, 15, 25, 45, 75]
filas = []
for k in ks:
    pred_j = KNNClasificador(k=k).predecir(sim_jaccard, y_train)
    pred_c = KNNClasificador(k=k).predecir(sim_coseno, y_train)
    filas.append({"k": k, "acc_jaccard": accuracy(y_test, pred_j), "acc_coseno": accuracy(y_test, pred_c)})

df_resultado = pd.DataFrame(filas)
print(df_resultado.to_string(index=False))
df_resultado.to_csv("resultados/barrido_k.csv", index=False)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(df_resultado["k"], df_resultado["acc_jaccard"], marker="o", label="k-NN + Jaccard")
ax.plot(df_resultado["k"], df_resultado["acc_coseno"], marker="o", label="k-NN + Coseno")
ax.set_xlabel("k")
ax.set_ylabel("accuracy")
ax.set_title("Accuracy vs k — MuchoCine (3 clases)")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("resultados/barrido_k.png", dpi=150)
print("\nGuardado en resultados/barrido_k.csv y resultados/barrido_k.png")
