# Documentación de la API de Recomendación (FastAPI)

Este script (`model_api.py`) levanta un servidor web local que actúa como interfaz entre el **Modelo Neuronal entrenado** y el **Frontend** (la página web). Su trabajo es recibir peticiones HTTP, consultar los datos en memoria y devolver respuestas en formato JSON.

## Configuración Inicial

### Dependencias

La API utiliza **FastAPI** por su velocidad y generación automática de documentación.

  * `fastapi` & `uvicorn`: Para el servidor web.
  * `joblib`: Para cargar el "cerebro" (el archivo `.joblib`).
  * `pandas` & `numpy`: Para manipulación de datos y vectores.
  * `tensorflow`: (Opcional) Para cargar el modelo Encoder si se quisiera procesar canciones nuevas que no están en el dataset.

### CORS (Cross-Origin Resource Sharing)

El script incluye una configuración de `CORSMiddleware`.

  * **Qué hace:** Permite que el Frontend (viviendo en `localhost:8080` o `5500`) pueda hacer peticiones a este Backend (viviendo en `localhost:8090`) sin que el navegador bloquee la conexión por seguridad.

-----

## Ciclo de Vida: El Evento `startup`

Cuando se ejecuta la API, ocurre lo siguiente antes de aceptar cualquier petición:

1.  **Carga de Artefactos:** La función `load_data()` busca el archivo `music_recommender_neural.joblib`.
2.  **Desempaquetado en Memoria (RAM):** Extrae y guarda en variables globales:
      * `df`: La base de datos de canciones (con nombres de clusters y features).
      * `embeddings_matrix`: La matriz numérica optimizada para cálculos rápidos.
      * `nombres_clusters`: El diccionario de traducción (ej: 0 -\> "Mainstream Hits").
3.  **Validación:** Verifica que los datos críticos (como la columna `cluster`) existan.

> **Nota de Rendimiento:** Al cargar todo en memoria al inicio, la API responde en milisegundos porque no tiene que leer el disco duro cada vez que alguien pide una canción.

-----

## Endpoints Disponibles

### 1\. Estado del Sistema

**`GET /health`**

  * **Función:** Diagnóstico. Verifica si la API está viva y si cargó los datos correctamente.
  * **Respuesta:** JSON con el estado "Funcionando", cantidad de filas cargadas y número de clusters.

### 2\. Buscar Canción

**`GET /search?song=...`**

  * **Función:** Busca coincidencias parciales por nombre. útil para el autocompletado en el frontend.
  * **Parámetros:** `song` (texto a buscar).
  * **Respuesta:** Lista de canciones que coinciden, incluyendo sus índices y características (Energy, Danceability) para pre-visualizar datos.

### 3\. Obtener Recomendaciones (Core)

**`GET /recommend/{song_name}`**

Es el endpoint principal. Realiza la magia de la recomendación.

**Parámetros (Query Params):**

  * `artist` (opcional): Para desambiguar si hay dos canciones con el mismo nombre (ej: "Hello" de Adele vs "Hello" de Lionel Richie).
  * `n` (default=10): Cantidad de canciones a recomendar.
  * `same_cluster` (default=True): Si es `True`, fuerza a que las recomendaciones pertenezcan al mismo "estilo musical" (Cluster) que la original.

**Flujo de Lógica Interna:**

1.  **Resolver:** Busca el índice de la canción en el DataFrame. Si hay duplicados, pide el artista.
2.  **Embedding:** Obtiene el vector numérico (Latent Space) de la canción elegida.
3.  **Distancia:** Calcula la **Distancia Euclidiana** entre ese vector y *todos* los demás vectores de la base de datos.
4.  **Ranking:** Ordena de menor a mayor distancia (más cerca = más parecido).
5.  **Filtrado:** Elimina la propia canción y aplica el filtro de Cluster si se solicitó.
6.  **Respuesta:** Construye un JSON con los datos de la canción original y la lista de recomendadas.

**Ejemplo de Respuesta JSON:**

```json
{
  "status": "ok",
  "original_song": {
    "song": "Shape of You",
    "nombre_cluster": "Mainstream Hits",
    "features": { "Danceability": 0.82, "Energy": 0.65 ... }
  },
  "recommendations": [
    {
      "song": "Despacito",
      "distance": 0.12,
      "features": { ... }
    },
    ...
  ]
}
```

-----

## Cómo ejecutar la API

Abrir la terminal en la carpeta donde está el script y el archivo `.joblib` y ejecuta:

```bash
uvicorn model_api:app --reload
```

  * `model_api`: Nombre del archivo Python.
  * `app`: Nombre de la instancia FastAPI dentro del archivo.
  * `--reload`: Reiniciar el servidor automáticamente si cambia el código (modo desarrollo).

La API estará disponible en: `http://127.0.0.1:8090`

Documentación interactiva (Swagger): `http://127.0.0.1:8090/docs`