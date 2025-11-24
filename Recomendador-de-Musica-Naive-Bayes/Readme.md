# Trabajo Práctico Machine Learning - Recomendador Musical Híbrido
# Grupo 1 - 25/11/2025

## Integrantes
- Macías, Juliana
- Cortés Cid, Francisco
- Moreno, Nahuel
- Teixido, Leonardo

---

# 1. Visión General del Sistema Híbrido

Este proyecto implementa un sistema de recomendación musical híbrido que combina la precisión de la **Similitud (KNN)** con la coherencia de los **Estilos Musicales (K-Means)**. El sistema se compone de tres partes interconectadas:

1.  **Entrenador (`train_model.py`):** Encargado de entrenar y guardar los modelos (artefactos).
2.  **API (`model_api.py`):** Servidor RESTful con FastAPI que procesa las solicitudes de recomendación.
3.  **Frontend (`index.html`):** Interfaz de usuario para la interacción web.

---

# 2. El Entrenador (`train_model.py`)

Este script realiza el preprocesamiento de los datos, el entrenamiento de los modelos y la persistencia de los artefactos.

## a. Features y Preprocesamiento

- **Features:** Utiliza **11 características numéricas** de audio (Danceability, Energy, Loudness, Tempo, Positiveness, etc.).
- **Escalado:** Las características se **estandarizan** con `StandardScaler` para asegurar la equidad en el cálculo de distancias.

## b. Componentes del Modelo

El sistema se basa en tres modelos:

### i. Agrupamiento de Estilos (K-Means)

- **Objetivo:** Clasificar las canciones en **6 tipos de música** distintos (Clusters).
- **Rol en el API:** Proporciona el ID de cluster para cada canción, permitiendo el filtrado de estilo.

### ii. Recomendador por Similitud (Nearest Neighbors - KNN)

- **Objetivo:** Encontrar las canciones más cercanas (similares) a una canción de entrada.
- **Métrica:** **Distancia Coseno** (`metric=cosine`) para medir la similitud de patrones de audio.

### iii. Persistencia

Todos los modelos (`knn_model`, `scaler`, `kmeans_model`, `naive_bayes_model`) y el DataFrame con los clusters asignados se guardan en el archivo **`music_recommender_with_clusters.joblib`**.

---

# 3. API RESTful (`model_api.py`)

La API carga los modelos en memoria al inicio y expone la lógica de recomendación a través de endpoints REST.

## a. Carga y Estabilidad

- **Startup:** La función `load_data` se ejecuta al inicio, leyendo el archivo `.joblib` desde la **ruta absoluta** para inicializar los modelos globales.
- **CORS:** Configurado para permitir la comunicación entre el puerto del *frontend* (`localhost:8080`) y el puerto de la API (`localhost:8090`).

## b. Endpoints Clave

| Ruta | Método | Descripción | Parámetros (Query) |
| :--- | :--- | :--- | :--- |
| **`/recommend/{song_name}`** | GET | **Ruta principal**. Genera recomendaciones, aplicando el filtro híbrido. | `artist` (str), `n` (int), `same_cluster` (bool) |
| **`/health`** | GET | Chequeo de estado. Confirma que los modelos y datos están cargados. | Ninguno |

## c. Lógica Híbrida

La función `recommend_by_index` utiliza el parámetro `same_cluster`. Si es `True`, las recomendaciones del KNN se **filtran** para mostrar solo las canciones que pertenecen al mismo estilo musical (Cluster) de la canción semilla.

---

# 4. Frontend (`index.html`)

La interfaz de usuario implementa la lógica necesaria para interactuar con la API y visualizar los resultados.

## a. Interfaz de Control

- **Filtro de Cluster:** El **checkbox** (`id="sameClusterCheckbox"`) permite al usuario controlar el filtro de estilo (`same_cluster`) que se envía al *backend*.

## b. Renderizado de Resultados

La función `renderOkPayload` procesa el JSON de la API y muestra la información de forma detallada:

- **Canción Semilla:** Título, Artista, características y su **Nombre/ID de Cluster**.
- **Recomendaciones:** Lista detallada que incluye la **Similitud** de KNN y la etiqueta del **Nombre/ID de Cluster** correspondiente.

---

# 5. Resumen de Tecnologías

| Componente | Tecnología | Propósito |
| :--- | :--- | :--- |
| **Modelos** | Python, Scikit-learn (KNN, K-Means, GaussianNB) | Núcleo de recomendación y clasificación. |
| **Backend** | FastAPI, Uvicorn | Servidor RESTful de alto rendimiento. |
| **Frontend** | HTML5, CSS3, JavaScript | Interfaz de usuario interactiva y asíncrona. |
```