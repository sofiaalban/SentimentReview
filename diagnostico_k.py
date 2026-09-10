"""Por qué Jaccard le saca más ventaja a Coseno a medida que k crece.

No es solo "accuracy sube con k" (eso es normal en cualquier k-NN) — la
pregunta es por qué el HUECO entre Jaccard y Coseno se abre con k en vez de
mantenerse parejo. Este script mide 3 cosas por separado para explicarlo con
números, no con intuición:

  1. Pureza de vecinos: de los k vecinos más cercanos, ¿qué fracción
     realmente comparte la etiqueta verdadera? (más alto = mejor señal)
  2. Empates: ¿cuántos de los "top-k" vecinos tienen exactamente la MISMA
     similitud? (Jaccard son razones de enteros -> muchos empates posibles;
     coseno son reales -> casi nunca hay empates exactos)
  3. Caída de similitud: qué tan parecido es el vecino #25 al vecino #1 en
     cada métrica (si caen poco, agregar más vecinos casi no mete ruido).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from src.texto import tokenize, split_train_test
from src.vectorize import Vectorizador
from src.similitud import jaccard_similitud, coseno_similitud

df = pd.read_csv("data/muchocine_reviews.csv")
df_train, df_test = split_train_test(df, "sentimiento", test_size=0.2, seed=42)
train_tok = [tokenize(t) for t in df_train["review_body"]]
test_tok = [tokenize(t) for t in df_test["review_body"]]
y_train = np.asarray(df_train["sentimiento"].tolist())
y_test = np.asarray(df_test["sentimiento"].tolist())

vec = Vectorizador(min_df=2).fit(train_tok)
bin_train, bin_test = vec.transform_binaria(train_tok), vec.transform_binaria(test_tok)
tfidf_train, tfidf_test = vec.transform_tfidf(train_tok), vec.transform_tfidf(test_tok)

sim_jaccard = jaccard_similitud(bin_test, bin_train)
sim_coseno = coseno_similitud(tfidf_test, tfidf_train)

print("=" * 70)
print("1) PUREZA DE VECINOS — % de los k vecinos que comparten la etiqueta real")
print("=" * 70)
ks = [1, 5, 15, 25, 45, 75]
filas = []
for k in ks:
    pureza_j, pureza_c = [], []
    for i in range(len(y_test)):
        idx_j = np.argsort(sim_jaccard[i])[::-1][:k]
        idx_c = np.argsort(sim_coseno[i])[::-1][:k]
        pureza_j.append(np.mean(y_train[idx_j] == y_test[i]))
        pureza_c.append(np.mean(y_train[idx_c] == y_test[i]))
    filas.append({"k": k, "pureza_jaccard": np.mean(pureza_j), "pureza_coseno": np.mean(pureza_c)})
tabla_pureza = pd.DataFrame(filas)
print(tabla_pureza.to_string(index=False))

print()
print("=" * 70)
print("2) EMPATES — cuántos test docs tienen 2+ vecinos EXACTAMENTE con la")
print("   misma similitud en la posición límite del top-k (empate en el corte)")
print("=" * 70)
for k in ks:
    empates_j = empates_c = 0
    for i in range(len(y_test)):
        orden_j = np.sort(sim_jaccard[i])[::-1]
        orden_c = np.sort(sim_coseno[i])[::-1]
        if k < len(orden_j) and np.isclose(orden_j[k - 1], orden_j[k]):
            empates_j += 1
        if k < len(orden_c) and np.isclose(orden_c[k - 1], orden_c[k]):
            empates_c += 1
    print(f"  k={k:3d}  ->  Jaccard: {empates_j}/{len(y_test)} test docs con empate en el corte "
          f"({100*empates_j/len(y_test):.1f}%)   ·   Coseno: {empates_c}/{len(y_test)} ({100*empates_c/len(y_test):.1f}%)")

print()
print("=" * 70)
print("3) CAÍDA DE SIMILITUD — valor del vecino #1 vs #25, promedio sobre todo el test")
print("=" * 70)
top1_j = np.sort(sim_jaccard, axis=1)[:, -1].mean()
top25_j = np.sort(sim_jaccard, axis=1)[:, -25].mean()
top1_c = np.sort(sim_coseno, axis=1)[:, -1].mean()
top25_c = np.sort(sim_coseno, axis=1)[:, -25].mean()
print(f"  Jaccard: vecino #1 = {top1_j:.4f}   vecino #25 = {top25_j:.4f}   caída = {100*(top1_j-top25_j)/top1_j:.1f}%")
print(f"  Coseno:  vecino #1 = {top1_c:.4f}   vecino #25 = {top25_c:.4f}   caída = {100*(top1_c-top25_c)/top1_c:.1f}%")

# también: cuántos valores DISTINTOS de similitud existen en total (granularidad)
print()
print("Valores únicos de similitud en toda la matriz (granularidad de la métrica):")
print(f"  Jaccard: {len(np.unique(np.round(sim_jaccard, 6)))} valores distintos")
print(f"  Coseno:  {len(np.unique(np.round(sim_coseno, 6)))} valores distintos")

tabla_pureza.to_csv("resultados/diagnostico_pureza.csv", index=False)
print("\nGuardado en resultados/diagnostico_pureza.csv")
