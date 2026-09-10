"""Implementación propia de las dos métricas, vectorizada con matrices dispersas
para poder correr k-NN sobre miles de reseñas en segundos en vez de horas.
Las fórmulas son las mismas que en el razonamiento matemático — aquí solo se
calculan para todos los pares (test, train) a la vez en vez de un par a la vez.
"""
import numpy as np
from scipy.sparse import csr_matrix


def jaccard_similitud(bin_a: csr_matrix, bin_b: csr_matrix) -> np.ndarray:
    """D_J = 1 - |A∩B|/|A∪B|  →  acá devolvemos la similitud 1-D_J = |A∩B|/|A∪B|.

    |A∩B| entre dos vectores binarios es literalmente su producto punto
    (ambos en 1 en la misma posición). |A∪B| = |A| + |B| - |A∩B|.
    """
    interseccion = bin_a.dot(bin_b.T).toarray()
    tam_a = np.asarray(bin_a.sum(axis=1)).flatten()
    tam_b = np.asarray(bin_b.sum(axis=1)).flatten()
    union = tam_a[:, None] + tam_b[None, :] - interseccion
    union[union == 0] = 1  # dos textos vacíos: similitud 0 por convención
    return interseccion / union


def coseno_similitud(vec_a: csr_matrix, vec_b: csr_matrix) -> np.ndarray:
    """cos(A,B) = (A·B) / (||A|| ||B||)"""
    producto_punto = vec_a.dot(vec_b.T).toarray()
    norma_a = np.sqrt(np.asarray(vec_a.multiply(vec_a).sum(axis=1)).flatten())
    norma_b = np.sqrt(np.asarray(vec_b.multiply(vec_b).sum(axis=1)).flatten())
    denominador = norma_a[:, None] * norma_b[None, :]
    denominador[denominador == 0] = 1  # vector nulo (texto sin palabras del vocabulario): similitud 0
    return producto_punto / denominador
