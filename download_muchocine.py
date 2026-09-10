"""
Descarga y prepara el corpus MuchoCine (reseñas de cine en español) para el
mini-proyecto de Modelos Matemáticos para la IA (tema: reviews-sentimientos).

Fuente: us-lsi/muchocine (Hugging Face), 3872 reseñas de películas en
español con calificación de 1 a 5 estrellas. Corpus original de Cruz,
Troyano, Enríquez y Ortega (2008), recopilado de muchocine.net.

El dataset no trae la etiqueta "Positivo/Negativo/Neutro" directo, así que
se deriva de las estrellas:
    1-2 estrellas -> negativo
    3   estrellas -> neutro
    4-5 estrellas -> positivo

Uso:
    pip install -r requirements.txt
    python download_muchocine.py                  # descarga todo el corpus
    python download_muchocine.py --sample 600      # muestra chica para probar rápido

Salida:
    data/muchocine_reviews.csv
        columnas: review_summary, review_body, star_rating, sentimiento
"""
import argparse
import sys
from pathlib import Path


def estrellas_a_sentimiento(star_rating: int) -> str:
    if star_rating <= 2:
        return "negativo"
    if star_rating == 3:
        return "neutro"
    return "positivo"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sample", type=int, default=None, help="Tomar solo N reseñas (muestreo estratificado por clase) para pruebas rápidas de k-NN.")
    parser.add_argument("--seed", type=int, default=42, help="Semilla para el muestreo aleatorio.")
    parser.add_argument("--out", type=str, default="data/muchocine_reviews.csv", help="Ruta del CSV de salida.")
    args = parser.parse_args()

    try:
        from datasets import load_dataset, concatenate_datasets
    except ImportError:
        sys.exit("Falta la librería 'datasets'. Instálala con:\n    pip install -r requirements.txt\n")

    print("Descargando MuchoCine desde Hugging Face (us-lsi/muchocine)...")

    # us-lsi/muchocine todavía usa el formato viejo de "loading script" (un .py que arma
    # el dataset), y las versiones recientes de `datasets` ya no lo ejecutan por defecto.
    # Primera opción: la copia en Parquet que Hugging Face genera automáticamente para
    # casi todo dataset público — no ejecuta ningún código, solo lee archivos de datos.
    # Si por lo que sea esa rama no existe, caemos al script original con
    # trust_remote_code=True (esto sí ejecuta el código Python que trae el repo del
    # dataset — es un corpus académico conocido, pero es una decisión consciente de
    # confianza, no algo que hacer por defecto en cualquier dataset).
    ds = None
    errores = []
    try:
        ds = load_dataset("us-lsi/muchocine", revision="refs/convert/parquet")
    except Exception as e:
        errores.append(f"Vía Parquet auto-convertido: {e}")
        try:
            ds = load_dataset("us-lsi/muchocine", trust_remote_code=True)
        except Exception as e2:
            errores.append(f"Vía script original (trust_remote_code=True): {e2}")

    if ds is None:
        sys.exit(
            "No se pudo descargar el dataset por ninguna de las dos vías intentadas.\n"
            "- Si estás detrás de un proxy corporativo o en un entorno restringido, corre este\n"
            "  script en Google Colab o en tu máquina local en vez de un entorno sandboxeado.\n"
            "- Si el error menciona 'trust_remote_code', puede que tu versión de `datasets`\n"
            "  ya no soporte ejecutar scripts en absoluto; prueba `pip install \"datasets<4\"`.\n\n"
            + "\n".join(errores)
        )

    # El dataset puede venir en uno o varios splits (train/validation/test); los juntamos todos
    # porque para este ejercicio nos interesa el corpus completo, no una partición específica.
    splits = list(ds.keys())
    full = concatenate_datasets([ds[s] for s in splits]) if len(splits) > 1 else ds[splits[0]]
    df = full.to_pandas()

    # star_rating puede venir codificado como ClassLabel de HF (0-4) o como entero plano (1-5)
    # según la versión del dataset — normalizamos siempre a la escala real de 1 a 5 estrellas.
    if df["star_rating"].min() == 0:
        df["star_rating"] = df["star_rating"] + 1

    df["sentimiento"] = df["star_rating"].apply(estrellas_a_sentimiento)

    df["review_body"] = df["review_body"].astype(str).str.strip()
    df["review_summary"] = df["review_summary"].astype(str).str.strip()
    df = df[df["review_body"].str.len() > 0].reset_index(drop=True)

    if args.sample and args.sample < len(df):
        frac = args.sample / len(df)
        df = (
            df.groupby("sentimiento", group_keys=False)
            .sample(frac=frac, random_state=args.seed)
            .reset_index(drop=True)
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df[["review_summary", "review_body", "star_rating", "sentimiento"]].to_csv(out_path, index=False)

    print(f"\nListo. {len(df)} reseñas guardadas en {out_path}")
    print("\nDistribución por clase:")
    print(df["sentimiento"].value_counts())


if __name__ == "__main__":
    main()
