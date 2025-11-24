# Instructivo Recomendador de Música Híbrido

## 1) Estructura de Carpetas

```
📁 Recomendador-de-Musica-Redes-Neuronales/
├─📁 front_end/
│    └─ index.html     # Interfaz de Usuario (Frontend)
│
├─📁 model-python/
│    ├─ train_model.py # Script de entrenamiento (ML)
│    ├─ model_api.py   # Servidor API (FastAPI)
│    ├─ music_recommender_with_clusters.joblib # Artefacto generado (Modelo serializado)
│    │  (aquí se guardarán los artefactos del modelo entrenado)
│    ├─📁 colab/
│    	  ├─ train_model.ipynb
│    	  ├─ model_api.ipynb
│    
├─📁 documentacion/
│    📁├─ media/
│      ├─ Api_Recomendador.md
│      ├─ Ejemplos.md
│      ├─ Entrenamiento del Modelo.md
│      ├─ FontEnd.md
│      ├─ Instructivo.md
│
├── Readme.md # Esta documentación```
---

## 2) Entrenar el modelo

Desde el directorio `model-python` ejecutar:

```
python train_model.py
```

Esto generará los artefactos de entrenamiento (`*.joblib`) para no tener que correr el entrenador cada vez que se ejecute el `frontend/`.

---

## 3) Levantar el Frontend (HTML)

Desde el directorio `frontend/` ejecutar:

```
python -m http.server [puerto_frontend]
```
Debe existir como una entrada de `allow_origins` en `CORSMiddleware` como `http://localhost:[puerto_frontend]` en `model_api.py`, si no existe, agregar.

Correrá en:  
`[http://localhost:[puerto_frontend]](http://localhost:[puerto_frontend])`

Default: **8080**

## 4) Levantar el Backend (FastAPI)

Desde el directorio `model-python` ejecutar:

```
python -m uvicorn model_api:app --host 0.0.0.0 --port [puerto_backend] --reload
```

El `[puerto_backend]` debe definirse en `API_BASE` de `index.html` como `API_BASE = "http://localhost:[puerto_backend]"`


Correrá en:  
`[http://localhost:[puerto_backend]](http://localhost:[puerto_backend])`

Default: **8090**
