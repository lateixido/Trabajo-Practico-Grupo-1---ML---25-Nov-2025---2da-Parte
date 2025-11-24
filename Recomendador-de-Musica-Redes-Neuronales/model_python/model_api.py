from fastapi import FastAPI, Query
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics.pairwise import euclidean_distances
from fastapi.middleware.cors import CORSMiddleware
import os

# Intentamos importar tensorflow para cargar los modelos de Keras serializados
try:
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    from tensorflow.keras.models import Model
except ImportError:
    print("Advertencia: TensorFlow no está instalado. Si necesitas usar el encoder para nuevas canciones, fallará.")

app = FastAPI(title="API Recomendador Spotify Neural (Autoencoder + Embeddings)")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8080",
        "http://127.0.0.1:8080",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://localhost:5500",
        "http://127.0.0.1:5500",
        "null",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------- Globals --------
df = None
scaler = None
kmeans_model = None
encoder_model = None  # El modelo que comprime features -> embeddings
embeddings_matrix = None # La "base de datos vectorial" (numpy array)
nombres_clusters = {}

# Configuración
ID_COLS = ["song", "artist"]
# Estas deben coincidir con las usadas en el entrenamiento
FEATURE_COLS = [
    "Danceability", "Energy", "variance", "Tempo", "Loudness",
    "Acousticness", "Instrumentalness", "Speechiness", 
    "Positiveness", "Popularity", "Liveness"
]
# Columnas para mostrar en el JSON de respuesta
OUTPUT_COLS = ["Danceability", "Energy", "Positiveness", "Loudness", "Tempo"]

# Nombre del archivo generado por el script de entrenamiento neuronal
ARTIFACTS_PATH = "music_recommender_neural.joblib"


# ----------------------------------------
# Comienzo: Cargar datos y modelos
# ----------------------------------------
@app.on_event("startup")
def load_data():
    """
    Cargar los artefactos del modelo neuronal: Scaler, Encoder, Embeddings y DataFrame.
    """
    global df, scaler, kmeans_model, encoder_model, embeddings_matrix, nombres_clusters

    if not os.path.exists(ARTIFACTS_PATH):
        raise RuntimeError(f"El archivo de artefactos no existe: {ARTIFACTS_PATH}")

    print(f"Cargando artefactos desde {ARTIFACTS_PATH}...")
    artifacts = joblib.load(ARTIFACTS_PATH)
    
    # 1. Cargar objetos clave
    scaler = artifacts.get("scaler")
    kmeans_model = artifacts.get("kmeans_model")
    nombres_clusters = artifacts.get("nombres_clusters", {})
    encoder_model = artifacts.get("encoder_model") # Modelo Keras
    embeddings_matrix = artifacts.get("embeddings") # Matriz Numpy (Latent Space)
    
    # 2. Cargar el DataFrame (que ya incluye clusters y columnas de embedding)
    df = artifacts.get("dataframe_data")
    
    if df is None or scaler is None or embeddings_matrix is None:
        raise RuntimeError("Artefactos incompletos. Faltan datos críticos (df, scaler o embeddings).")
    
    # Validar columnas
    # Nota: 'cluster' y las columnas de embedding ya vienen en 'dataframe_data' según el script de entrenamiento
    if "cluster" not in df.columns:
        # Si por alguna razón no está, se re-asigna usando el modelo kmeans
        print("⚠️ Columna 'cluster' no encontrada, re-asignando...")
        # Esto requeriría tener las features originales en el df, se asumen que están.
        pass 

    print(f"✅ Datos Neurales cargados. Filas: {df.shape[0]}. Dimensión Embedding: {embeddings_matrix.shape[1]}")


# ---------------------------------------
# Ayudas
# ---------------------------------------
def _norm(s: str) -> str:
    return s.casefold().strip()

def resolver_cancion(song_name: str):
    """Resuelve canción por nombre."""
    name_mask = df["song"].astype(str).str.casefold().str.strip() == _norm(song_name)
    if not name_mask.any():
        return {
            "status": "No_Encontrado",
            "message": "Cancion no encontrada. Por favor trate con otro nombre."
        }

    idxs = df.index[name_mask].tolist()
    if len(idxs) > 1:
        artists = df.loc[idxs, "artist"].dropna().unique().tolist()
        return {
            "status": "Artista_Necesario",
            "message": f"Titulo ambiguo '{song_name}'. Seleccione artista.",
            "options": {"artists": sorted(artists)}
        }

    return {"status": "single", "index": idxs[0]}

def resolver_cancion_con_artista(song_name: str, artist: str):
    """Resuelve canción por nombre y artista."""
    name_mask = df["song"].astype(str).str.casefold().str.strip() == _norm(song_name)
    if not name_mask.any():
        return {"status": "No_Encontrado", "message": "Cancion no encontrada."}

    artist_mask = df["artist"].astype(str).str.casefold().str.strip() == _norm(artist)
    both_mask = name_mask & artist_mask
    
    if not both_mask.any():
        # Título existe, artista no
        idxs = df.index[name_mask].tolist()
        artists = df.loc[idxs, "artist"].dropna().unique().tolist()
        return {
            "status": "Artista_Necesario",
            "message": f"Artista '{artist}' no coincide con '{song_name}'.",
            "options": {"artists": sorted(artists)}
        }

    idxs = df.index[both_mask].tolist()
    return {"status": "single", "index": idxs[0]}

def get_embedding_vector(row_idx: int):
    """
    Obtiene el vector de embedding para un índice dado.
    Como ya se tienen los embeddings pre-calculados en 'embeddings_matrix',
    es una búsqueda directa O(1).
    """
    return embeddings_matrix[row_idx].reshape(1, -1)

def recomendacion_por_embeddings(row_idx: int, top_k: int = 20, same_cluster: bool = True):
    """
    Calcula recomendaciones usando Distancia Euclidiana en el espacio latente.
    """
    # 1. Obtener vector semilla
    target_vec = get_embedding_vector(row_idx)
    seed_cluster_id = int(df.at[row_idx, "cluster"])
    seed_cluster_name = nombres_clusters.get(seed_cluster_id, f"Cluster {seed_cluster_id}")

    # 2. Calcular distancias contra TODOS los vectores (Vectorized operation = Fast)
    # embeddings_matrix shape: (N_songs, Latent_Dim)
    # target_vec shape: (1, Latent_Dim)
    dists = euclidean_distances(target_vec, embeddings_matrix).flatten()

    # 3. Ordenar índices por distancia (ascendente)
    # argsort devuelve los índices que ordenarían el array
    sorted_indices = dists.argsort()

    recs = []
    seen = set()
    
    # Iteramos sobre los índices ordenados
    # Empezamos desde 1 para saltar la propia canción (distancia 0)
    for i in sorted_indices:
        if i == row_idx:
            continue

        # Filtro de Cluster
        current_cluster = int(df.at[i, "cluster"])
        if same_cluster and current_cluster != seed_cluster_id:
            continue
        
        # Filtro de duplicados
        key = (df.at[i, "song"], df.at[i, "artist"])
        if key in seen:
            continue
        seen.add(key)
        
        # Construir objeto de recomendación
        rec_row = df.iloc[i]
        
        
        features_dict = {}
        for col in OUTPUT_COLS:
            val = rec_row.get(col, None) # Usar .get por seguridad
            if val is not None:
                features_dict[col] = float(val)

        recs.append({
            "song": rec_row["song"],
            "artist": rec_row["artist"],
            "cluster_id": current_cluster,
            "nombre_cluster": nombres_clusters.get(current_cluster, f"Cluster {current_cluster}"),
            "distance": float(dists[i]), # Distancia euclidiana
            "similarity": float(1.0 / (1.0 + dists[i])), # Conversión simple a similitud 0-1
            **features_dict
        })

        if len(recs) >= top_k:
            break

    # Datos de la canción semilla
    seed_row = df.iloc[row_idx]
    seed_features = {}
    for col in OUTPUT_COLS:
        val = seed_row.get(col, None)
        if val is not None:
            seed_features[col] = float(val)

    return {
        "original_song": {
            "song": seed_row["song"],
            "artist": seed_row["artist"],
            "cluster_id": seed_cluster_id,
            "nombre_cluster": seed_cluster_name,
            "features": seed_features
        },
        "recommendations": recs
    }


# ---------------------------------------
# Endpoint REST
# ---------------------------------------
@app.get("/recommend/{song_name}")
def recommend(song_name: str,
              artist: str | None = Query(default=None),
              n: int = Query(default=10, ge=1, le=50),
              same_cluster: bool = Query(default=True)):
    """
    Genera recomendaciones usando el Autoencoder Neural.
    """
    if artist is None:
        res = resolver_cancion(song_name)
    else:
        res = resolver_cancion_con_artista(song_name, artist)

    if res["status"] in ["Artista_Necesario", "No_Encontrado"]:
        return res

    # Usamos la nueva lógica de embeddings
    payload = recomendacion_por_embeddings(res["index"], top_k=n, same_cluster=same_cluster)
    return {"status": "ok", **payload}


@app.get("/search")
def search(song: str):
    """Busca coincidencias (igual que antes)."""
    name_norm = _norm(song)
    mask = df["song"].astype(str).str.casefold().str.strip() == name_norm
    # Obtenemos info básica
    matches = df.loc
    
@app.get("/")
def root():
    return {"message": "API de Recomendación de Música con Red Neuronal (Autoencoder)", "status": "active"}


@app.get("/health")
def health_check():
    return {
        "Grupo": "1",
        "Materia": "Machine Learning",
        "Trabajo Practico": "Recomendador de Música con Red Neuronal (Autoencoder)",
        "Estado": "Funcionando",
        "Dataset Cargado": df is not None,
        "Escalador Cargado": scaler is not None,
        "Modelo KMeans Cargado": kmeans_model is not None,
        "Columnas de Características": len(FEATURE_COLS),
        "Clusters": len(nombres_clusters),
        "Filas del Dataset": int(df.shape[0]) if df is not None else 0
    }