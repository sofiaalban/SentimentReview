# Mini-proyecto — Reviews-Sentimientos

Modelos Matemáticos para la IA · clasificación de reseñas en Positivo / Neutro / Negativo.

## Dataset: MuchoCine

3872 reseñas de películas en español (`us-lsi/muchocine`, Hugging Face), con
calificación de 1 a 5 estrellas. Se deriva la etiqueta de 3 clases así:

| Estrellas | Sentimiento |
|-----------|-------------|
| 1-2       | negativo    |
| 3         | neutro      |
| 4-5       | positivo    |

## Cómo correr el script

**Importante:** este script necesita descargar datos de `huggingface.co`. Si
lo corres dentro de una sesión de Claude Code sobre este vault, esa conexión
está bloqueada por política de red del entorno — el script falla con un
mensaje claro explicándolo, no con un error críptico. Para que sí descargue,
corre estos mismos archivos en **Google Colab** o en tu máquina local (ahí
`huggingface.co` es accesible sin problema).

```bash
pip install -r requirements.txt
python download_muchocine.py                # corpus completo (3872 reseñas)
python download_muchocine.py --sample 600    # muestra chica, útil para probar
                                              # k-NN con Jaccard antes de escalar
                                              # (Jaccard compara por pares, es
                                              # más lento que coseno con vocabulario
                                              # global)
```

Salida: `data/muchocine_reviews.csv` con columnas
`review_summary, review_body, star_rating, sentimiento`.

## Alcance algorítmico planeado

1. **k-NN + distancia de Jaccard** — reseñas como conjuntos de palabras.
2. **k-NN + similitud de coseno sobre TF-IDF** — reseñas como vectores ponderados.
3. **Centroide supervisado ("k-means modificado") + coseno sobre TF-IDF** —
   equivalente al *Rocchio / Nearest Centroid Classifier*: un centroide por
   clase (promedio de los vectores TF-IDF de esa clase), clasificación por
   cercanía al centroide más próximo. No aplica a Jaccard porque un centroide
   es un promedio y Jaccard opera sobre conjuntos, no vectores.

Comparaciones a reportar entre (1)/(2) vs (3): accuracy por clase (ojo
particular a "neutro", que es la clase con más riesgo de que el centroide
mezcle reseñas muy distintas entre sí), interpretabilidad (palabras de mayor
peso por centroide) y tiempo de clasificación.

## Implementación y resultados

Código en `src/` (tokenización + stopwords, vectorización binaria/TF-IDF,
similitudes, k-NN, centroide, evaluación) y `run_experiment.py` como script
principal. Nada de `sklearn.neighbors` ni `sklearn.metrics`: las fórmulas de
Jaccard, coseno, k-NN, el centroide y la matriz de confusión están escritas a
mano (vectorizadas con `scipy.sparse` solo por velocidad, no delegadas a una
librería que ya las calcule).

```bash
python run_experiment.py --k 15                 # corpus completo
python run_experiment.py --sample 600 --k 15     # muestra chica para iterar rápido
python barrido_k.py                              # accuracy vs k, para elegir k
```

### Clasificar una reseña nueva (para el video/demo)

`run_experiment.py` mide accuracy separando el corpus en train/test — no
clasifica texto inventado. Para eso está `predict_review.py`: usa las 3871
reseñas como base de conocimiento completa y clasifica cualquier texto nuevo
que le pases, mostrando además los 5 vecinos reales más parecidos.

```bash
python predict_review.py                        # modo interactivo, ideal para grabar
python predict_review.py --texto "Tu reseña acá" # un solo disparo
```

Salida en `resultados/`: matrices de confusión (PNG), métricas por clase,
tiempos de predicción, palabras top por centroide, y `reporte.txt` con todo
junto.

### Resultado principal (corpus completo, 3097 train / 774 test, k=15)

| Modelo                          | Accuracy | Tiempo (774 predicciones) |
|----------------------------------|---------:|---------------------------:|
| k-NN + Jaccard                   |   57.4%  | 0.52s |
| k-NN + Coseno/TF-IDF              |   51.3%  | 0.32s |
| Centroide supervisado + Coseno    |   **59.9%**  | **0.01s** |

**Hallazgo que contradice la intuición del razonamiento teórico:** en este
corpus, k-NN con Jaccard superó a k-NN con Coseno — es consistente en todo
un barrido de k de 1 a 75 (ver `resultados/barrido_k.png`), no es un
accidente de k=15. Hipótesis: las reseñas de MuchoCine son muy largas (481
palabras en promedio, hasta 4585), y con un vocabulario de ~33 mil palabras
el TF de cada término se vuelve minúsculo (1 entre cientos), diluyendo el
vector TF-IDF entre miles de dimensiones de peso parecido — mientras que
Jaccard, al ignorar la frecuencia y mirar solo presencia/ausencia con
denominador local, no sufre ese efecto de dilución. El argumento teórico del
proyecto (coseno > Jaccard) se sostiene para textos cortos con vocabulario
compartido chico; para reseñas largas como estas, el resultado empírico va
en la otra dirección — vale la pena mostrar esto en la presentación como
matiz, no como error.

El centroide (mismo TF-IDF que a k-NN le fue peor) termina ganando de todas
formas — la hipótesis original era que promediar iba a perjudicar
específicamente a "neutro" por mezclar reseñas heterogéneas; **no se
confirmó**: el F1 de "neutro" con centroide (0.45) fue mejor que con k-NN +
coseno (0.35) y comparable a k-NN + Jaccard (0.46). Lo que sí se sostiene:
"neutro" es, en las tres técnicas, la clase con peor desempeño — se confunde
con negativo y positivo casi por igual, justo lo que uno esperaría de una
categoría intermedia. Y el centroide es ~30-50x más rápido en predicción
porque compara contra 3 puntos en vez de contra las 3097 reseñas de train.
