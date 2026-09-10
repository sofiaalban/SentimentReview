"""Los dos algoritmos del proyecto:

- KNNClasificador: instancia-basado, memoriza todo el train, vota entre los
  k vecinos más similares. Recibe la matriz de similitud ya calculada
  (Jaccard o coseno) para no acoplarse a una métrica en particular.

- CentroideClasificador: la versión "k-means supervisado" (Rocchio /
  Nearest Centroid) — un centroide TF-IDF por clase, se predice por
  cercanía de coseno al centroide más próximo. No tiene sentido para
  Jaccard (los conjuntos no se promedian), así que solo trabaja con
  vectores TF-IDF.
"""
from collections import Counter
from typing import List, Sequence

import numpy as np
from scipy.sparse import csr_matrix

from .similitud import coseno_similitud


class KNNClasificador:
    def __init__(self, k: int = 5):
        self.k = k

    def predecir(self, matriz_similitud: np.ndarray, y_train: Sequence[str]) -> List[str]:
        y_train = np.asarray(y_train)
        predicciones = []
        for fila in matriz_similitud:
            vecinos_idx = np.argsort(fila)[::-1][: self.k]
            vecinos_labels = y_train[vecinos_idx]
            vecinos_sim = fila[vecinos_idx]

            conteo = Counter(vecinos_labels)
            max_votos = max(conteo.values())
            empatados = [clase for clase, votos in conteo.items() if votos == max_votos]

            if len(empatados) == 1:
                predicciones.append(empatados[0])
            else:
                # desempate: la clase empatada cuya similitud acumulada entre
                # los k vecinos sea mayor (no es al azar, sigue midiendo "cercanía").
                suma_sim = {c: vecinos_sim[vecinos_labels == c].sum() for c in empatados}
                predicciones.append(max(suma_sim, key=suma_sim.get))
        return predicciones


class CentroideClasificador:
    """Rocchio / Nearest Centroid — el "k-means modificado" del proyecto:
    un centroide por clase construido con las etiquetas (no iterado como en
    k-means clásico), clasificación por coseno al centroide más cercano.
    """

    def __init__(self):
        self.clases: List[str] = []
        self.centroides: csr_matrix = None

    def fit(self, tfidf_train: csr_matrix, y_train: Sequence[str]):
        y_train = np.asarray(y_train)
        self.clases = sorted(set(y_train))
        filas = []
        for clase in self.clases:
            mask = y_train == clase
            centroide = np.asarray(tfidf_train[mask].mean(axis=0)).flatten()
            filas.append(centroide)
        self.centroides = csr_matrix(np.vstack(filas))
        return self

    def predecir(self, tfidf_test: csr_matrix) -> List[str]:
        sim = coseno_similitud(tfidf_test, self.centroides)  # (n_test, n_clases)
        idx_pred = np.argmax(sim, axis=1)
        return [self.clases[i] for i in idx_pred]

    def top_terminos_por_clase(self, vocab: dict, n: int = 15) -> dict:
        """Las n palabras de mayor peso en el centroide de cada clase —
        el vocabulario "prototipo" que k-NN nunca resume."""
        idx_a_palabra = {i: w for w, i in vocab.items()}
        resultado = {}
        centroides_densos = self.centroides.toarray()
        for i, clase in enumerate(self.clases):
            top_idx = np.argsort(centroides_densos[i])[::-1][:n]
            resultado[clase] = [(idx_a_palabra[j], round(float(centroides_densos[i, j]), 4)) for j in top_idx]
        return resultado
