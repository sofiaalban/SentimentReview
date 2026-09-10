- Barrido de k = 1 hasta k = 75: Jaccard, en reseñas larguísimas con vocabulario de ~33 mil palabras, no le importa la frecuencia — solo cuenta palabras compartidas. En un corpus de reseñas de cine, hay _muchísimos_ documentos que comparten una cantidad moderada de vocabulario genérico de cine (film, historia, director, actuación...) con cualquier reseña nueva, así que en vez de un "ganador claro" tienes una meseta ancha de candidatos todos razonablemente parecidos. Coseno con TF-IDF, en cambio, premia fuerte a quien comparte las palabras _raras_ — así que cuando encuentra una reseña con overlap real en vocabulario específico, esa gana por mucho, y todo lo demás cae rápido a casi nada.

### Por qué eso hace que k les afecte distinto

- **Con una curva empinada (Coseno):** casi toda la señal útil está en los primeros vecinos. Agrandar k no suma información nueva — solo mete vecinos cada vez menos relacionados, diluyendo el voto con ruido. Por eso Coseno se aplana rápido (~k=25-45) y no mejora más.
- **Con una curva plana (Jaccard):** no hay "un" vecino claramente mejor, hay muchos moderadamente buenos. Con k=1 estás apostando por uno de ellos casi al azar — de hecho lo confirmé: al cortar en k=75, el **13.7% de las reseñas de test tienen empates exactos** de similitud en Jaccard (contra solo 0.4% en Coseno), porque Jaccard es una razón de enteros y por eso tiene mucha menos granularidad (24,825 valores distintos de similitud vs 75,664 en Coseno). Agrandar k no mete ruido — promedia el voto entre varios candidatos igual de válidos y elimina esa aleatoriedad del vecino único. Por eso Jaccard sigue mejorando hasta k≈25.

---


### 1. k-means clásico: 3 amigos adivinando dónde están los grupos

Imagina un festival con miles de personas repartidas en un campo, sin ninguna organización visible, y tienes que adivinar dónde están los 3 grupos de amigos que sí existen, pero sin que nadie te diga quién es amigo de quién. Eso es k-means: **no tienes etiquetas, tienes que descubrir la estructura**.

El algoritmo:

1. Tiras 3 banderas al azar en el campo (centroides iniciales).
2. Cada persona "se une" a la bandera más cercana.
3. Cada bandera camina hasta el centro exacto (el promedio de posición) de la gente que se le unió.
4. Repites 2-3 hasta que las banderas dejan de moverse.

Es prueba y error iterativo — nadie le dice a las banderas dónde pararse, ellas lo averiguan solas moviéndose hacia el promedio una y otra vez. Por eso es **no supervisado**: nunca usa una respuesta correcta, solo la geometría de los puntos.

### 2. Cómo lo volvimos supervisado: nos saltamos la adivinanza

Nuestro caso es distinto — **ya sabemos** que cada reseña de MuchoCine es Positiva, Negativa o Neutra (viene de las estrellas). No necesitamos que 3 banderas caminen a ciegas probando dónde pararse: podemos ponerlas directamente en el centro exacto del grupo correcto desde el primer intento, porque ya sabemos quién pertenece a cada grupo.

Eso es literalmente lo que hace el código — nada de iterar, nada de repetir, un solo cálculo:

def fit(self, tfidf_train, y_train):
    self.clases = sorted(set(y_train))
    filas = []
    for clase in self.clases:
        mask = y_train == clase          # <- AQUÍ está la supervisión
        centroide = np.asarray(tfidf_train[mask].mean(axis=0)).flatten()
        filas.append(centroide)
    self.centroides = csr_matrix(np.vstack(filas))

`mask = y_train == clase` es la línea que convierte todo el algoritmo en supervisado: es la etiqueta real, puesta por un humano (la calificación de estrellas), la que decide qué reseñas se promedian juntas. El k-means clásico nunca ve `y_train` — ni siquiera existe esa variable en su versión original. Nuestra versión la usa como el ingrediente principal.

**La metáfora:** en vez de armar el retrato robot de un sospechoso promediando rasgos de fotos "que se parecen entre sí" (adivinando cuáles se parecen, como haría k-means), la policía ya tiene 1000 fotos correctamente archivadas bajo "Positivo" — solo promedias esas 1000 caras y obtienes un retrato robot: el rostro compuesto de "así se ve típicamente una reseña positiva". Haces lo mismo para Negativo y Neutro. Tres retratos robot, uno por categoría, construidos de un solo golpe porque el archivo ya venía ordenado.

### 3. La matemática exacta

Para la clase $c$ (Positivo, Negativo o Neutro), con $D_c$ = todas las reseñas de entrenamiento de esa clase:

centroide⃗c=1∣Dc∣∑d∈Dctfidf⃗d\vec{centroide}_c = \frac{1}{|D_c|}\sum_{d \in D_c} \vec{tfidf}_dcentroidec​=∣Dc​∣1​d∈Dc​∑​tfidf​d​

Es el promedio, coordenada por coordenada, de todos los vectores TF-IDF de esa clase — si "malo" pesa 0.3 en una reseña negativa y 0.5 en otra, el centroide de Negativo tiene 0.4 en la coordenada de "malo". Es la misma matemática del centro de gravedad físico, aplicada a un vector de miles de dimensiones en vez de a 3 coordenadas espaciales.

Para clasificar una reseña nueva, ni siquiera hace falta código nuevo — reutilizamos exactamente la misma función de similitud de coseno que ya usa k-NN:

def predecir(self, tfidf_test):
    sim = coseno_similitud(tfidf_test, self.centroides)  # (n_test, n_clases)
    idx_pred = np.argmax(sim, axis=1)
    return [self.clases[i] for i in idx_pred]

y^=arg⁡max⁡c  cos⁡(tfidf⃗nueva,centroide⃗c)\hat{y} = \arg\max_{c} \; \cos(\vec{tfidf}_{nueva}, \vec{centroide}_c)y^​=argcmax​cos(tfidf​nueva​,centroidec​)

Se mide el ángulo entre la reseña nueva y cada uno de los 3 retratos robot, y gana el que tenga el ángulo más chico (más parecido).

### 4. Por qué coseno es la única opción posible aquí (no es elección, es obligación matemática)

Un centroide es un **promedio**, y promediar solo tiene sentido en un espacio vectorial donde sumar y dividir estén definidos. Repasa las dos representaciones que construimos:

- **Jaccard usa conjuntos** — {"malo", "guión", "flojo"}. No existe una operación matemática estándar de "promediar conjuntos" — ¿qué significaría sumar dos conjuntos de palabras y dividir entre 2? No hay una definición sin inventarse una regla nueva. Por eso `CentroideClasificador.fit()` en el código solo acepta `tfidf_train`, nunca la matriz binaria.
- **Coseno usa vectores TF-IDF** — números reales en cada coordenada. Sumar y dividir entre n está perfectamente definido. Es la única de las dos representaciones que permite construir un centroide.

Y dentro de los vectores, coseno (en vez de, digamos, distancia euclidiana al centroide) sigue siendo la elección correcta por la misma razón que ya justificamos en la teoría: normaliza por la magnitud, así que una reseña larga y una corta con el mismo "sabor" de palabras caen cerca en ángulo aunque tengan pesos totales distintos. Euclidiana penalizaría a una reseña simplemente por ser más larga, no por decir cosas distintas.

### 5. Por qué es una buena opción para este proyecto específicamente

- Reutiliza exactamente la misma representación TF-IDF y la misma función `coseno_similitud` que ya construimos para k-NN — cero infraestructura nueva, sacamos más provecho al mismo trabajo.
- Extiende k-means, que es literalmente uno de los dos algoritmos vistos en clase, en vez de traer algo desconocido — es un puente natural entre lo que la profesora enseñó y lo que el proyecto necesita.
- Da un tercer ángulo de comparación genuinamente distinto: basado en instancias (k-NN memoriza todo) vs. basado en prototipos (centroide resume cada clase en un punto) — buen contraste conceptual para la sustentación.

### 6. Por qué terminó siendo el más rápido — matemática directa, no casualidad

k-NN + coseno, para clasificar una reseña, la compara contra las **3097** reseñas de entrenamiento, una por una, y luego vota. El centroide la compara contra **3** puntos — un retrato robot por clase — y ya.

k-NN: 774 resen˜as de test×3097 comparaciones≈2.4 millones\text{k-NN: } 774 \text{ reseñas de test} \times 3097 \text{ comparaciones} \approx 2.4 \text{ millones}k-NN: 774 resen˜as de test×3097 comparaciones≈2.4 millones

Centroide: 774×3=2322 comparaciones\text{Centroide: } 774 \times 3 = 2322 \text{ comparaciones}Centroide: 774×3=2322 comparaciones

Mil veces menos comparaciones matemáticamente — por eso 0.01s contra 0.32s. La metáfora: en vez de hojear las 3097 fotos del archivo una por una buscando la más parecida (k-NN), comparas contra 3 retratos robot ya armados de antemano. El trabajo pesado (armar los retratos) se hizo una sola vez en `fit()`; en el momento de clasificar, ya no hay archivo que hojear.

### 7. Por qué terminó teniendo el mejor accuracy — la conexión con lo que ya descubrimos

Esto se conecta directo con el diagnóstico que hicimos la vez pasada. Ya sabemos que las reseñas de MuchoCine son larguísimas (481 palabras en promedio) y que eso hace que **cada vector TF-IDF individual quede "diluido"** — mucho ruido, señal débil, tal como vimos que le pasaba a k-NN+coseno.

Promediar 1000+ reseñas de una clase para construir el centroide es, matemáticamente, una operación de **cancelación de ruido** — la misma idea detrás de por qué en estadística confías más en el promedio de 1000 mediciones que en una sola medición. Cada reseña individual tiene "ruido" propio (palabras idiosincráticas de esa película, de ese crítico, de ese día), pero ese ruido es aleatorio y distinto en cada reseña — al promediar 1000 de ellas, el ruido de cada una tiende a cancelarse entre sí, mientras que la señal real (las palabras que genuinamente diferencian "positivo" de "negativo" en el idioma) se refuerza porque aparece consistentemente en todas.

k-NN nunca hace esa cancelación: compara la reseña nueva contra reseñas individuales, una por una, cada una con todo su ruido propio intacto — por más que vote entre k=15 vecinos, sigue mirando puntos ruidosos uno por uno, no un resumen limpio.

**La metáfora final, cerrando el círculo:** un testigo individual (una reseña) puede describir mal a un sospechoso por los nervios, la luz, su memoria — ruido. Pero el retrato robot armado promediando la descripción de mil testigos que sí vieron a la misma clase de persona (mil reseñas positivas) cancela los errores individuales y deja solo los rasgos que de verdad se repiten. Por eso el retrato robot terminó siendo más confiable que preguntarle a testigos uno por uno — no porque sea un algoritmo "mejor" en abstracto, sino porque en reseñas tan largas y ruidosas como las de MuchoCine, promediar ayuda más de lo que cuesta perder el detalle individual.