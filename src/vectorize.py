"""Construye las representaciones vectoriales de los textos:
- matriz binaria (para Jaccard, ver conjuntos de palabras)
- matriz TF-IDF (para similitud de coseno)

El vocabulario y el IDF se calculan SOLO sobre el set de entrenamiento (para
no filtrar información del test) y las fórmulas son exactamente las que se
plantearon en el razonamiento matemático del proyecto:

    TF(palabra, texto) = veces que aparece / total de palabras del texto
    IDF(palabra)       = ln(N_train / textos_de_train_que_la_contienen)
"""
from collections import Counter
from typing import Dict, List, Sequence

import numpy as np
from scipy.sparse import csr_matrix


class Vectorizador:
    def __init__(self, min_df: int = 2):
        self.min_df = min_df
        self.vocab: Dict[str, int] = {}
        self.idf: np.ndarray = None
        self.n_train_docs: int = 0

    def fit(self, docs_tokenizados: Sequence[List[str]]):
        df_counter = Counter()
        for tokens in docs_tokenizados:
            df_counter.update(set(tokens))

        # descartamos palabras rarísimas (aparecen en < min_df textos): no aportan
        # nada a Jaccard/coseno y solo inflan el vocabulario y el tiempo de cómputo.
        vocab_words = [w for w, df in df_counter.items() if df >= self.min_df]
        self.vocab = {w: i for i, w in enumerate(vocab_words)}
        self.n_train_docs = len(docs_tokenizados)

        df_vec = np.array([df_counter[w] for w in vocab_words], dtype=float)
        self.idf = np.log(self.n_train_docs / df_vec)
        return self

    def _matriz_conteos(self, docs_tokenizados: Sequence[List[str]]) -> csr_matrix:
        rows, cols, data = [], [], []
        for i, tokens in enumerate(docs_tokenizados):
            conteo = Counter(t for t in tokens if t in self.vocab)
            for palabra, c in conteo.items():
                rows.append(i)
                cols.append(self.vocab[palabra])
                data.append(c)
        n_docs = len(docs_tokenizados)
        return csr_matrix((data, (rows, cols)), shape=(n_docs, len(self.vocab)), dtype=float)

    def transform_binaria(self, docs_tokenizados: Sequence[List[str]]) -> csr_matrix:
        """Presencia/ausencia de cada palabra del vocabulario — para Jaccard."""
        m = self._matriz_conteos(docs_tokenizados)
        m.data[:] = 1.0
        return m

    def transform_tfidf(self, docs_tokenizados: Sequence[List[str]]) -> csr_matrix:
        """TF x IDF — para similitud de coseno."""
        conteos = self._matriz_conteos(docs_tokenizados)
        largo_doc = np.asarray(conteos.sum(axis=1)).flatten()
        largo_doc[largo_doc == 0] = 1  # evita 0/0 en textos que quedaron vacíos tras filtrar vocab
        tf = conteos.multiply(1.0 / largo_doc[:, None]).tocsr()
        tfidf = tf.multiply(self.idf[None, :]).tocsr()
        return tfidf
