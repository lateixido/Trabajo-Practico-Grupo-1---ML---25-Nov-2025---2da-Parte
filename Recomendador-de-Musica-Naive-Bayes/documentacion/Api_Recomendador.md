# API de Recomendación de Música Híbrida (FastAPI)

Este script (`model_api.py`) implementa un servicio web RESTful utilizando **FastAPI** para exponer el modelo de recomendación entrenado (`music_recommender_with_clusters.joblib`). Su principal función es tomar una canción de entrada y devolver canciones similares, aplicando opcionalmente un filtro de estilo (Cluster).

## 1. Arquitectura y Configuración

### a. Inicialización y CORS

* **FastAPI (`app`):** Crea la instancia principal de la aplicación.
* **CORS Middleware:** Se configura **CORSMiddleware** para permitir que el *frontend* (ej., corriendo en `localhost:8080`) pueda comunicarse con el *backend* (ej., corriendo en `localhost:8090`). Esto evita errores de seguridad de origen cruzado.

### b. Variables Globales y Artefactos

Se definen variables globales (`df`, `knn_model`, `scaler`, etc.) que almacenarán los modelos cargados una vez que la aplicación se inicie.

* **`FEATURE_COLS` (11 columnas):** La lista completa de características de audio utilizadas para la similitud (ej., `Energy`, `Loudness`).
* **`OUTPUT_COLS` (4 columnas):** Un subconjunto de características que se devuelve al *frontend* para una visualización concisa.
* **`ARTIFACTS_PATH`:** La ruta absoluta que garantiza la carga correcta del archivo `.joblib`.

---

## 2. Carga del Modelo en el Inicio (*Startup*)

La función `load_data` es crítica; se ejecuta **solo una vez** cuando el servidor se inicia (`@app.on_event("startup")`).

* **Persistencia:** Utiliza `joblib.load(ARTIFACTS_PATH)` para deserializar el archivo entrenado (`.joblib`).
* **Inicialización:** Asigna los artefactos a las variables globales:
    * **`knn_model`:** El modelo de **Nearest Neighbors** para calcular la similitud.
    * **`scaler`:** El **StandardScaler** para preprocesar las nuevas canciones de entrada.
    * **`kmeans_model`** y **`CLUSTER_NAMES`:** El modelo de clustering y los nombres de los 6 estilos de música identificados.
    * **`df` (`track_index`):** El DataFrame índice que contiene todas las canciones y sus IDs de cluster asignados.
* **Validación:** El código verifica si el archivo existe y si todos los modelos y datos clave se cargaron correctamente. Si el `startup` falla, la API no servirá peticiones de forma estable.

---

## 3. Funciones Auxiliares (*Helpers*)

Estas funciones internas se encargan de la lógica de preprocesamiento y búsqueda.

* `_norm(s)`: Normaliza *strings* (minúsculas, sin espacios) para búsquedas de canciones y artistas insensibles a mayúsculas.
* `resolver_cancion` / `resolver_cancion_con_artista`: Manejan la lógica de búsqueda de la canción de entrada en el `df` global. Su objetivo es devolver el **índice** de la canción semilla.
    * Si el título es ambiguo, devuelve un *status* **`need_artist`** con opciones.
    * Si no se encuentra, devuelve **`not_found`**.
* `crea_vector_caracteristicas(row_idx)`: Toma el índice de una canción, extrae sus 11 *features*, aplica el **`scaler`** global y la prepara en formato `csr_matrix` para el KNN.

### `recomendacion_por_indice` (Lógica Híbrida Principal)

Esta es la función central que implementa la recomendación:

1.  Obtiene el vector de features (`q_vec`) y el **ID del cluster de la canción semilla**.
2.  Utiliza `knn_model.kneighbors()` para encontrar las canciones más cercanas.
3.  **Filtro Híbrido:** Itera sobre los vecinos. Si el parámetro **`same_cluster`** es `True` (por defecto), **descarta** cualquier recomendación cuyo `cluster_id` sea diferente al de la canción semilla. 
4.  Construye el *payload* JSON final con detalles de la canción original y la lista filtrada de recomendaciones, incluyendo la **similaridad**, el **ID del cluster** y el **nombre del cluster**.

---

## 4. Endpoints REST (API Pública)

La API expone tres rutas principales:

### a. GET `/recommend/{song_name}`

Esta es la ruta principal que utiliza el *frontend* para obtener recomendaciones.

* **Parámetros de Query:**
    * `artist`: Opcional, para desambiguar el título.
    * `n`: Cantidad de recomendaciones a devolver (por defecto, 10).
    * `same_cluster`: **Booleano** que activa/desactiva el filtro de estilo (cluster).
* **Flujo:**
    1.  Resuelve la canción y el artista (maneja `not_found` o `need_artist`).
    2.  Llama a `recomendacion_por_indice` para obtener las recomendaciones, aplicando el filtro `same_cluster`.

### b. GET `/search`

Ruta auxiliar utilizada para buscar coincidencias de títulos de canciones y devolver la lista de artistas disponibles en caso de ambigüedad.

### c. GET `/health` y `/`

* **`/health`:** Ruta de diagnóstico. Devuelve el estado de la API, confirmando si el DataFrame y todos los modelos (`knn_model`, `scaler`, `kmeans_model`, etc.) se cargaron correctamente.
* **`/`:** Mensaje de bienvenida simple.