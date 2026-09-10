"""Tokenización y split train/test. Sin librerías de NLP: solo lo necesario
para pasar de texto crudo a listas de palabras, tal como se planteó en el
razonamiento matemático del mini-proyecto.
"""
import re
from typing import List

import numpy as np
import pandas as pd

_TOKEN_RE = re.compile(r"[^\W\d_]+", re.UNICODE)  # letras (incluye acentos/ñ), sin dígitos ni puntuación


_TABLA_ACENTOS = str.maketrans("áéíóúü", "aeiouu")


def _quitar_acentos(palabra: str) -> str:
    """está/esta, según/segun -> misma forma, para que la lista de stopwords
    (y el vocabulario en general) no trate como palabras distintas dos formas
    que solo difieren en tilde. Traducción explícita en vez de NFD genérico
    porque NFD descompone la ñ en n + tilde, y "año" no puede volverse "ano"."""
    return palabra.translate(_TABLA_ACENTOS)

# Stopwords en español: artículos, preposiciones, pronombres, conjunciones y
# formas comunes de ser/estar/haber. Se filtran porque en reseñas largas (los
# textos de MuchoCine promedian ~480 palabras) su frecuencia bruta es tan alta
# que el TF-IDF no las castiga lo suficiente, y terminan dominando tanto el
# vocabulario como las palabras top de cada centroide sin aportar nada al
# sentimiento del texto.
STOPWORDS_ES = frozenset("""
a al algo algunas algunos ante antes como con contra cual cuando de del desde
donde durante e el ella ellas ellos en entre era erais eramos eran eres es esa
esas ese eso esos esta estaba estabais estabamos estaban estar estas este
esto estos estoy fue fuimos fue ser fui ha habia habeis habia han hasta hay
la las le les lo los mas me mi mis mismo mucho muy nada ni no nos nosotros
nosotras o os otra otras otro otros para pero poco por porque que quien se
sea sean segun sera seran si sido siendo sin sobre sois somos son soy su sus
suya suyas suyo suyos te tenia tenido tiene tienen todo todos tu tus un una
uno unos y ya yo
""".split())


def tokenize(text: str) -> List[str]:
    """Minúsculas, sin acentos, sin puntuación ni números ni stopwords."""
    crudos = (_quitar_acentos(t) for t in _TOKEN_RE.findall(text.lower()))
    return [t for t in crudos if t not in STOPWORDS_ES and len(t) > 1]


def split_train_test(df: pd.DataFrame, label_col: str, test_size: float = 0.2, seed: int = 42):
    """Split estratificado hecho a mano (mantiene la proporción de cada clase
    en train y test) para no depender de scikit-learn solo por esto.
    """
    rng = np.random.default_rng(seed)
    test_idx = []
    for _, grupo in df.groupby(label_col):
        n_test = max(1, round(len(grupo) * test_size))
        idx = rng.choice(grupo.index.to_numpy(), size=n_test, replace=False)
        test_idx.extend(idx)
    test_idx = set(test_idx)
    df_test = df.loc[df.index.isin(test_idx)].reset_index(drop=True)
    df_train = df.loc[~df.index.isin(test_idx)].reset_index(drop=True)
    return df_train, df_test
