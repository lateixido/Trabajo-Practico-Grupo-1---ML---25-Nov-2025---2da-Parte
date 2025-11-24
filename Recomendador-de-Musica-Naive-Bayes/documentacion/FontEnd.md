# 💻 Interfaz de Usuario (Frontend HTML/JavaScript)

El archivo HTML proporcionado es la interfaz de usuario (UI) que permite a los usuarios interactuar con el **API de Recomendación de Música Híbrida** construido con FastAPI y Python. La lógica principal se ejecuta en JavaScript (JS) para gestionar la comunicación asíncrona con el *backend* y renderizar los resultados.

## 1. 🖼️ Estructura y Estilo (HTML y CSS)

El *frontend* utiliza HTML para la estructura y CSS simple (definido en el bloque `<style>`) para un diseño limpio y funcional.

* **Formulario de Búsqueda (`<form id="form">`):** Contiene los controles de entrada para la solicitud.
* **Controles Clave:**
    * **Canción de Referencia (`<input id="song">`):** Campo principal para la búsqueda.
    * **Selector de Artista (`<select id="artist">`):** Aparece dinámicamente para resolver ambigüedades.
    * **Filtro Híbrido (`<input type="checkbox" id="sameClusterCheckbox">`):** Permite al usuario elegir si desea **restringir las recomendaciones** al mismo estilo musical (Cluster) de la canción semilla. 
* **Contenedor de Resultados (`<div id="output">`):** Donde se muestran la canción semilla, sus características y la lista de recomendaciones.

## 2. 🧠 Lógica de Comunicación (JavaScript)

El bloque `<script>` gestiona el flujo de trabajo, desde la interacción del usuario hasta la visualización de los resultados.

### a. Configuración y Constantes

* `API_BASE = "http://localhost:8090"`: Define la dirección del servidor de la API de FastAPI. **Es crucial que este puerto coincida con el puerto donde se ejecuta el *backend*.**
* `pct(v)` y `fmt(n)`: Funciones de ayuda para formatear números como porcentajes y valores decimales fijos, facilitando la lectura de *features* y similitud.

### b. Función Central: `fetchRecommendations`

Esta función maneja la comunicación con el *backend*:

1.  **Construcción de URL:** Lee el nombre de la canción, el artista elegido y el estado del *checkbox* `sameClusterCheckbox`.
2.  **Envío del Parámetro Híbrido:** Incluye el parámetro `same_cluster` en la URL de la API, controlando si el *backend* debe aplicar el filtro K-Means/Cluster.
3.  **Llamada Asíncrona:** Usa la API `fetch()` para enviar la solicitud `GET` al servidor de FastAPI (`/recommend/{song_name}`).
4.  **Manejo de Errores HTTP:** Captura errores de red (códigos 4xx o 5xx).

### c. Flujo de Control (`form.addEventListener("submit")`)

Este manejador de eventos define la lógica de interacción completa:

1.  **Validación de Entrada:** Verifica si se ingresó un nombre de canción.
2.  **Lógica de Desambiguación:** Si la API responde con `status: "need_artist"`, el JS **oculta los resultados** y llama a `populateArtists` para mostrar el *dropdown* con las opciones de artistas, esperando una nueva solicitud.
3.  **Visualización de Errores:** Si la API responde con `status: "not_found"`, muestra un mensaje de error claro.
4.  **Procesamiento Exitoso:** Si la respuesta es `status: "ok"`, llama a `renderOkPayload`.

### d. Renderizado de Resultados (`renderOkPayload`)

Esta función procesa la respuesta JSON de la API y genera el HTML dinámico:

1.  **Canción Original:** Muestra el título, artista, y sus **4 características clave** (`Danceability`, `Energy`, `Positiveness`, `Loudness`). Lo más importante es que etiqueta la canción con su **nombre y ID de Cluster** (ej., "Cluster 3").
2.  **Recomendaciones:** Itera sobre la lista de canciones recomendadas. Cada elemento muestra:
    * Título y Artista.
    * Píldora de **Similitud** (calculada como 1 - Distancia Coseno).
    * Píldora de **Cluster** (Nombre e ID), confirmando que la canción coincide con el estilo de la canción semilla (si el filtro estaba activo).