# Documentación del Frontend (Interfaz de Usuario)

Este archivo (`index.html`) proporciona una interfaz web limpia y moderna para interactuar con la API de recomendación. Está construido con **HTML, CSS y JavaScript puro (Vanilla JS)**, sin necesidad de frameworks complejos como React o Angular, lo que lo hace ligero y fácil de desplegar.

## Características Principales

1.  **Búsqueda Interactiva:** Permite al usuario ingresar el nombre de una canción.
2.  **Manejo de Ambigüedad:** Si existen varias canciones con el mismo nombre (ej: "Hello"), la interfaz despliega automáticamente un selector de artistas.
3.  **Filtros Inteligentes:** Incluye un *checkbox* para activar/desactivar el filtro de Cluster (para explorar música similar o diversa).
4.  **Visualización de Datos:** Muestra métricas clave de audio (`Danceability`, `Energy`, etc.) tanto para la canción original como para las recomendaciones.
5.  **Feedback Visual:** Indicadores de carga ("Loading..."), mensajes de error y etiquetas de colores para los Clusters.

## Estructura del Código

### 1\. Estilos (CSS)

El diseño utiliza **CSS Variables** (`:root`) para facilitar el cambio de temas y colores.

  * **Diseño:** Grid y Flexbox para organizar las tarjetas de canciones.
  * **Componentes:**
      * `.card`: Contenedores con sombra suave para los resultados.
      * `.pill`: Etiquetas redondeadas para mostrar el Cluster y el porcentaje de similitud.
      * `code`: Estilo monoespaciado para resaltar los valores numéricos (ej: `0.85`).

### 2\. Lógica (JavaScript)

El script gestiona la comunicación asíncrona con el Backend.

#### Configuración

```javascript
const API_BASE = "http://localhost:8090";
```

  * **Importante:** Esta variable define a dónde se enviarán las peticiones. Debe coincidir con la dirección y puerto donde corre el `model_api.py`.

#### Flujo de la Búsqueda

1.  **Evento Submit:** Captura el formulario y previene la recarga de la página.
2.  **Fetch API:** Llama al endpoint `/recommend` pasando el nombre de la canción y el estado del checkbox de cluster.
3.  **Manejo de Estados:**
      * *Status "ok":* Llama a `renderOkPayload()` para dibujar los resultados.
      * *Status "Artista\_Necesario":* Muestra el dropdown (`select`) con los artistas disponibles.
      * *Status "No\_Encontrado":* Muestra un mensaje de error amigable.

#### Renderizado Dinámico (`renderOkPayload`)

Esta función recibe el JSON de la API y construye el HTML en tiempo real.

  * Muestra el nombre del Cluster "Humano" (ej: *Mainstream Hits*) en lugar del número ID.
  * Calcula el porcentaje de similitud basado en la distancia inversa.
  * Dibuja los detalles técnicos (`Energy`, `Danceability`) para que el usuario entienda por qué se recomendó esa canción.

## 🚀 Cómo usar el Frontend

### Prerrequisitos

Asegurarse de que la API (`model_api.py`) esté ejecutándose. Por ejemplo:

```bash
uvicorn model_api:app --port 8090
```

*(Nota: El puerto en el HTML está configurado en 8090. Asegurarse de que coincida con la API).*

### Ejecución

Hay dos opciones para abrir el archivo:

1.  **Opción Recomendada (VS Code):**

      * Instala la extensión "Live Server".
      * Haz clic derecho en `index.html` -\> "Open with Live Server".
      * Esto evita problemas de seguridad (CORS) que a veces ocurren al abrir archivos locales.

2.  **Opción Directa:**

      * Simplemente haz doble clic en el archivo `index.html` para abrirlo en el navegador (Chrome, Edge, Firefox).
      * *Nota:* Como la API tiene configurado `CORS`, debería funcionar correctamente incluso abriéndolo como archivo local.

-----

## Guía Visual de la Interfaz

  * **Píldora Violeta:** Indica el Cluster (Grupo Musical) al que pertenece la canción.
  * **Píldora Azul:** Indica qué tan similar es matemáticamente la recomendación (100% es idéntica).
  * **Datos en Gris:** Muestra los valores normalizados de las características de audio.

-----

### Solución de Problemas Comunes

  * **Error "Failed to fetch":** Significa que el Frontend no puede ver a la API. Verifica que la API esté corriendo y que el puerto en `const API_BASE` sea el correcto.
  * **No aparecen artistas en el dropdown:** Verificar que la base de datos (`.csv`) tenga la columna `artist` correctamente cargada.