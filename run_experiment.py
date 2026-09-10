"""Corre los 3 experimentos del mini-proyecto sobre MuchoCine:

    1) k-NN + distancia de Jaccard      (reseñas como conjuntos de palabras)
    2) k-NN + similitud de coseno/TF-IDF (reseñas como vectores ponderados)
    3) Centroide supervisado ("k-means modificado") + coseno/TF-IDF

y guarda en resultados/: matrices de confusión (PNG), métricas por clase,
tiempos de predicción y las palabras top de cada centroide.

Uso:
    pip install -r requirements.txt
    python run_experiment.py --sample 1200 --k 15
"""
import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from src.texto import tokenize, split_train_test
from src.vectorize import Vectorizador
from src.similitud import jaccard_similitud, coseno_similitud
from src.clasificadores import KNNClasificador, CentroideClasificador
from src.evaluacion import matriz_confusion, metricas_por_clase, accuracy, graficar_matriz_confusion


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", type=str, default="data/muchocine_reviews.csv")
    parser.add_argument("--sample", type=int, default=None, help="Submuestra estratificada del corpus (útil para iterar rápido).")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--k", type=int, default=15, help="k para k-NN.")
    parser.add_argument("--min-df", type=int, default=2, help="Descarta palabras que aparecen en menos de N textos de train.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="resultados")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)

    print(f"Cargando {args.csv}...")
    df = pd.read_csv(args.csv)

    if args.sample and args.sample < len(df):
        frac = args.sample / len(df)
        df = df.groupby("sentimiento", group_keys=False).sample(frac=frac, random_state=args.seed).reset_index(drop=True)

    df_train, df_test = split_train_test(df, "sentimiento", test_size=args.test_size, seed=args.seed)
    print(f"Train: {len(df_train)}  ·  Test: {len(df_test)}")
    print(df_train["sentimiento"].value_counts(), "\n")

    print("Tokenizando...")
    train_tok = [tokenize(t) for t in df_train["review_body"]]
    test_tok = [tokenize(t) for t in df_test["review_body"]]
    y_train = df_train["sentimiento"].tolist()
    y_test = df_test["sentimiento"].tolist()
    clases = sorted(set(y_train))

    print("Construyendo vocabulario y matrices (binaria + TF-IDF)...")
    vec = Vectorizador(min_df=args.min_df).fit(train_tok)
    print(f"Vocabulario: {len(vec.vocab)} palabras\n")

    bin_train = vec.transform_binaria(train_tok)
    bin_test = vec.transform_binaria(test_tok)
    tfidf_train = vec.transform_tfidf(train_tok)
    tfidf_test = vec.transform_tfidf(test_tok)

    resumen = []

    # --- 1) k-NN + Jaccard ---------------------------------------------------
    print(f"[1/3] k-NN (k={args.k}) + Jaccard...")
    t0 = time.perf_counter()
    sim_jaccard = jaccard_similitud(bin_test, bin_train)
    pred_jaccard = KNNClasificador(k=args.k).predecir(sim_jaccard, y_train)
    t_jaccard = time.perf_counter() - t0

    acc_jaccard = accuracy(y_test, pred_jaccard)
    cm_jaccard = matriz_confusion(y_test, pred_jaccard, clases)
    met_jaccard = metricas_por_clase(cm_jaccard, clases)
    graficar_matriz_confusion(cm_jaccard, f"k-NN + Jaccard (k={args.k}, acc={acc_jaccard:.2f})", out_dir / "cm_knn_jaccard.png")
    print(f"  accuracy = {acc_jaccard:.3f}  ·  {t_jaccard:.2f}s para {len(df_test)} predicciones\n")
    resumen.append({"modelo": "k-NN + Jaccard", "accuracy": acc_jaccard, "tiempo_s": t_jaccard})

    # --- 2) k-NN + Coseno/TF-IDF ----------------------------------------------
    print(f"[2/3] k-NN (k={args.k}) + Coseno/TF-IDF...")
    t0 = time.perf_counter()
    sim_coseno = coseno_similitud(tfidf_test, tfidf_train)
    pred_coseno = KNNClasificador(k=args.k).predecir(sim_coseno, y_train)
    t_coseno = time.perf_counter() - t0

    acc_coseno = accuracy(y_test, pred_coseno)
    cm_coseno = matriz_confusion(y_test, pred_coseno, clases)
    met_coseno = metricas_por_clase(cm_coseno, clases)
    graficar_matriz_confusion(cm_coseno, f"k-NN + Coseno (k={args.k}, acc={acc_coseno:.2f})", out_dir / "cm_knn_coseno.png")
    print(f"  accuracy = {acc_coseno:.3f}  ·  {t_coseno:.2f}s para {len(df_test)} predicciones\n")
    resumen.append({"modelo": "k-NN + Coseno", "accuracy": acc_coseno, "tiempo_s": t_coseno})

    # --- 3) Centroide supervisado ("k-means modificado") + Coseno -------------
    print("[3/3] Centroide supervisado (Rocchio) + Coseno...")
    t0 = time.perf_counter()
    centroide = CentroideClasificador().fit(tfidf_train, y_train)
    pred_centroide = centroide.predecir(tfidf_test)
    t_centroide = time.perf_counter() - t0

    acc_centroide = accuracy(y_test, pred_centroide)
    cm_centroide = matriz_confusion(y_test, pred_centroide, clases)
    met_centroide = metricas_por_clase(cm_centroide, clases)
    graficar_matriz_confusion(cm_centroide, f"Centroide + Coseno (acc={acc_centroide:.2f})", out_dir / "cm_centroide.png")
    print(f"  accuracy = {acc_centroide:.3f}  ·  {t_centroide:.2f}s para {len(df_test)} predicciones\n")
    resumen.append({"modelo": "Centroide (k-means supervisado) + Coseno", "accuracy": acc_centroide, "tiempo_s": t_centroide})

    top_terminos = centroide.top_terminos_por_clase(vec.vocab, n=15)

    # --- guardar reporte ------------------------------------------------------
    df_resumen = pd.DataFrame(resumen)
    df_resumen.to_csv(out_dir / "resumen_accuracy_tiempos.csv", index=False)

    with open(out_dir / "reporte.txt", "w", encoding="utf-8") as f:
        f.write(f"Corpus: {len(df)} reseñas totales · train={len(df_train)} · test={len(df_test)}\n")
        f.write(f"Vocabulario (min_df={args.min_df}): {len(vec.vocab)} palabras\n")
        f.write(f"k usado en k-NN: {args.k}\n\n")

        f.write("=== Resumen ===\n")
        f.write(df_resumen.to_string(index=False) + "\n\n")

        for nombre, cm, met in [
            ("k-NN + Jaccard", cm_jaccard, met_jaccard),
            ("k-NN + Coseno", cm_coseno, met_coseno),
            ("Centroide + Coseno", cm_centroide, met_centroide),
        ]:
            f.write(f"=== {nombre} ===\n")
            f.write("Matriz de confusión:\n")
            f.write(cm.to_string() + "\n")
            f.write("\nMétricas por clase:\n")
            f.write(met.to_string() + "\n\n")

        f.write("=== Palabras de mayor peso por centroide (top 15) ===\n")
        for clase, terminos in top_terminos.items():
            f.write(f"\n{clase}:\n")
            for palabra, peso in terminos:
                f.write(f"  {palabra:20s} {peso}\n")

    print(f"Reporte completo guardado en {out_dir}/reporte.txt")
    print(f"Matrices de confusión (PNG) guardadas en {out_dir}/")
    print("\n=== Resumen ===")
    print(df_resumen.to_string(index=False))


if __name__ == "__main__":
    main()
