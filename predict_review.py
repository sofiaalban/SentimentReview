"""Clasifica una reseña NUEVA (que tú escribas) contra las 3871 reseñas
reales de MuchoCine — pensado para grabar el video de demo.

A diferencia de run_experiment.py (que mide accuracy separando train/test
del mismo corpus), este script usa TODO el corpus etiquetado como base de
conocimiento y clasifica un texto que no está en ningún lado del CSV: el
caso de uso real de "reseña nueva → algoritmo".

Uso:
    python predict_review.py
        (modo interactivo: te pide una reseña, la clasifica, te pregunta
        si quieres probar otra — ideal para grabar en vivo)

    python predict_review.py --texto "Esta película me encantó, actuaciones espectaculares"
        (modo de un solo disparo, útil para un demo con texto ya preparado)
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from src.texto import tokenize
from src.vectorize import Vectorizador
from src.similitud import jaccard_similitud, coseno_similitud
from src.clasificadores import KNNClasificador, CentroideClasificador

K = 15


def cargar_base(csv_path: str, min_df: int):
    print(f"Cargando y vectorizando {csv_path} (todo el corpus, sin split)...")
    df = pd.read_csv(csv_path)
    tokens = [tokenize(t) for t in df["review_body"]]
    y = df["sentimiento"].tolist()

    vec = Vectorizador(min_df=min_df).fit(tokens)
    bin_base = vec.transform_binaria(tokens)
    tfidf_base = vec.transform_tfidf(tokens)
    centroide = CentroideClasificador().fit(tfidf_base, y)

    print(f"Listo: {len(df)} reseñas, vocabulario de {len(vec.vocab)} palabras.\n")
    return df, vec, tokens, y, bin_base, tfidf_base, centroide


def clasificar(texto: str, vec, y, bin_base, tfidf_base, centroide, df):
    tok_nueva = [tokenize(texto)]

    bin_nueva = vec.transform_binaria(tok_nueva)
    tfidf_nueva = vec.transform_tfidf(tok_nueva)

    sim_jaccard = jaccard_similitud(bin_nueva, bin_base)
    sim_coseno = coseno_similitud(tfidf_nueva, tfidf_base)

    pred_jaccard = KNNClasificador(k=K).predecir(sim_jaccard, y)[0]
    pred_coseno = KNNClasificador(k=K).predecir(sim_coseno, y)[0]
    pred_centroide = centroide.predecir(tfidf_nueva)[0]

    print("=" * 60)
    print(f"Reseña nueva: \"{texto}\"")
    print(f"Palabras después de tokenizar (sin stopwords): {tok_nueva[0]}")
    print("=" * 60)
    print(f"  k-NN + Jaccard (k={K})      ->  {pred_jaccard}")
    print(f"  k-NN + Coseno/TF-IDF (k={K}) ->  {pred_coseno}")
    print(f"  Centroide + Coseno          ->  {pred_centroide}")
    print()

    print(f"Los {min(5,K)} vecinos más parecidos según coseno (de las 3871 reseñas reales):")
    vecinos_idx = sim_coseno[0].argsort()[::-1][:5]
    for idx in vecinos_idx:
        resumen = df.iloc[idx]["review_summary"]
        etiqueta = df.iloc[idx]["sentimiento"]
        similitud = sim_coseno[0][idx]
        print(f"  [{similitud:.3f}] ({etiqueta:9s}) {resumen}")
    print()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", type=str, default="data/muchocine_reviews.csv")
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--texto", type=str, default=None, help="Clasifica este texto y termina (sin modo interactivo).")
    args = parser.parse_args()

    df, vec, tokens, y, bin_base, tfidf_base, centroide = cargar_base(args.csv, args.min_df)

    if args.texto:
        clasificar(args.texto, vec, y, bin_base, tfidf_base, centroide, df)
        return

    print("Modo interactivo. Escribe una reseña de cine y presiona Enter.")
    print("Escribe 'salir' para terminar.\n")
    while True:
        texto = input(">> Reseña nueva: ").strip()
        if texto.lower() in ("salir", "exit", "quit", ""):
            print("Listo.")
            break
        clasificar(texto, vec, y, bin_base, tfidf_base, centroide, df)


if __name__ == "__main__":
    main()
