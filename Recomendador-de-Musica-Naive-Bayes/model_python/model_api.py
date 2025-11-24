from fastapi import FastAPI, Query
import pandas as pd
import numpy as np
import joblib
from scipy.sparse import csr_matrix
from fastapi.middleware.cors import CORSMiddleware
import os

app = FastAPI(title="API Recomendador Spotify Híbrido (KNN + Clusters)")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8080",
        "http://127.0.0.1:8080",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://localhost:5500",  # <-- Puerto común para Live Server
        "http://127.0.0.1:5500",
        "null",  # <-- para file:// origins
        # Agrega otros puertos de desarrollo si los usas (e.g., 5173, 3000)
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------- Globals --------
df = None
knn_model = None
scaler = None
kmeans_model = None
naive_bayes_model = None
CLUSTER_NAMES = {}

# Columnas esperadas y configuraciones para el nuevo modelo
ID_COLS = ["song", "artist"]
FEATURE_COLS = [
    "Danceability",
    "Energy",
    "variance",
    "Tempo",
    "Loudness",
    "Acousticness",
    "Instrumentalness",
    "Speechiness",
    "Positiveness",
    "Popularity",
    "Liveness",
]
# Columnas que se mostrarán en la respuesta JSON (subconjunto de FEATURE_COLS)
OUTPUT_COLS = ["Danceability", "Energy", "Positiveness", "Loudness"]

# La ruta del archivo de artefactos debe coincidir con el nombre guardado
ARTIFACTS_PATH = "music_recommender_with_clusters.joblib"


# ---------------------------
# Startup: load data & models
# ---------------------------
@app.on_event("startup")
def load_data():
    """
    Carga todos los artefactos entrenados (KNN, Scaler, KMeans, NaiveBayes, DataFrame)
    """
    global df, knn_model, scaler, kmeans_model, naive_bayes_model, nombres_clusters

    if not os.path.exists(ARTIFACTS_PATH):
        raise RuntimeError(f"El archivo de artefactos no existe: {ARTIFACTS_PATH}")

    artifacts = joblib.load(ARTIFACTS_PATH)
    
    # 1. Cargar modelos y objetos clave
    knn_model = artifacts.get("knn_model")
    scaler = artifacts.get("scaler")
    kmeans_model = artifacts.get("kmeans_model")
    naive_bayes_model = artifacts.get("naive_bayes_model")
    nombres_clusters = artifacts.get("nombres_clusters", {})
    
    # 2. Cargar el DataFrame índice (que ahora incluye la columna 'cluster')
    df = artifacts.get("track_index")
    
    if df is None or knn_model is None or scaler is None or kmeans_model is None:
        raise RuntimeError("Artefactos incompletos. Faltan modelos o el DataFrame 'track_index'.")
    
    # Asegurarse de que el DF cargado tiene las columnas esperadas, incluyendo 'cluster'
    required_cols = ID_COLS + FEATURE_COLS + ["cluster"]
    missing_df_cols = [c for c in required_cols if c not in df.columns]
    if missing_df_cols:
         raise RuntimeError(f"El DataFrame cargado (track_index) está incompleto. Faltan: {missing_df_cols}")
        
    print(f"✅ Datos y modelos cargados. Filas: {df.shape[0]}. Features: {len(FEATURE_COLS)}. Clusters: {len(nombres_clusters)}")


# ---------------------------------------
# Helpers
# ---------------------------------------
def _norm(s: str) -> str:
    """Normaliza strings a minúsculas y sin espacios iniciales/finales."""
    return s.casefold().strip()

def resolver_cancion(song_name: str):
    """Resuelve canción por nombre. Devuelve 'need_artist' si hay ambigüedad."""
    name_mask = df["song"].astype(str).str.casefold().str.strip() == _norm(song_name)
    if not name_mask.any():
        return {
            "status": "No_Encontrado",
            "message": "Cancion no encontrada. Por favor trate con otro nombre o verifique lo ingresado."
        }

    idxs = df.index[name_mask].tolist()
    if len(idxs) > 1:
        artists = (
            df.loc[idxs, "artist"]
            .dropna()
            .astype(str)
            .drop_duplicates()
            .tolist()
        )
        return {
            "status": "Artista_Necesario",
            "message": f"Titulo de canción ambiguo '{song_name}'. Por favor, seleccione el artista.",
            "options": {"artists": sorted(artists)}
        }

    return {"status": "single", "index": idxs[0]}

def resolver_cancion_con_artista(song_name: str, artist: str):
    """Resuelve canción por nombre y artista."""
    name_mask = df["song"].astype(str).str.casefold().str.strip() == _norm(song_name)
    if not name_mask.any():
        return {
            "status": "No_Encontrado",
            "message": "Cancion no encontrada. Por favor trate con otro nombre o verifique lo ingresado."
        }

    artist_mask = df["artist"].astype(str).str.casefold().str.strip() == _norm(artist)
    both_mask = name_mask & artist_mask
    if not both_mask.any():
        # título existe pero artista no coincide: ofrecer dropdown de artistas correctos
        idxs = df.index[name_mask].tolist()
        artists = (
            df.loc[idxs, "artist"]
            .dropna()
            .astype(str)
            .drop_duplicates()
            .tolist()
        )
        return {
            "status": "Artista_Necesario",
            "message": f"Artista '{artist}' no encontrado para la canción '{song_name}'. Por favor, seleccione un artista valido.",
            "options": {"artists": sorted(artists)}
        }

    # Solo toma el primer índice si hay duplicados (manejo simplificado)
    idxs = df.index[both_mask].tolist()
    return {"status": "single", "index": idxs[0]}

def crea_vector_caracteristicas(row_idx: int) -> csr_matrix:
    """Construye el vector de features (11 features) escalado para el KNN."""
    # Usamos FEATURE_COLS (las 11 columnas)
    row = df.loc[row_idx, FEATURE_COLS].to_numpy(dtype=float).reshape(1, -1)
    row_scaled = scaler.transform(row)
    return csr_matrix(row_scaled)

def recomendacion_por_indice(row_idx: int, top_k: int = 50, same_cluster: bool = True):
    """Genera recomendaciones por índice, con filtro opcional por cluster."""
    q_vec = crea_vector_caracteristicas(row_idx)
    seed_cluster_id = int(df.at[row_idx, "cluster"]) 
    seed_cluster_name = nombres_clusters.get(seed_cluster_id, f"Cluster {seed_cluster_id}")
    
    # Pedimos más vecinos (ej: top_k + 50) para tener suficiente margen para filtrar
    distances, indices = knn_model.kneighbors(q_vec, n_neighbors=top_k + 50) 
    
    recs = []
    seen = set()

    for d, i in zip(distances[0], indices[0]):
        if i == row_idx:
            continue
        
        # ⚠️ Lógica de Filtrado Híbrido por Cluster
        current_cluster = int(df.at[i, "cluster"])
        if same_cluster and current_cluster != seed_cluster_id:
            continue
        
        # Filtrar duplicados (song, artist)
        key = (df.at[i, "song"], df.at[i, "artist"])
        if key in seen:
            continue
        seen.add(key)

        rec = df.iloc[i]
        recs.append({
            "song": rec["song"],
            "artist": rec["artist"],
            "cluster_id": current_cluster,
            "nombre_cluster": nombres_clusters.get(current_cluster, f"Cluster {current_cluster}"),
            "similarity": float(max(0.0, min(1.0, 1.0 - float(d)))),
            # Incluimos las 4 features clave para inspección en el frontend
            **{col: float(rec[col]) for col in OUTPUT_COLS} 
        })
        
        if len(recs) == top_k:
            break

    seed = df.iloc[row_idx]
    return {
        "original_song": {
            "song": seed["song"],
            "artist": seed["artist"],
            "cluster_id": seed_cluster_id,
            "nombre_cluster": seed_cluster_name,
            "features": {col: float(seed[col]) for col in OUTPUT_COLS}
        },
        "recommendations": recs
    }


# ---------------------------------------
# REST endpoints
# ---------------------------------------
@app.get("/recommend/{song_name}")
def recommend(song_name: str,
              artist: str | None = Query(default=None),
              n: int = Query(default=50, ge=1, le=50),
              # Nuevo parámetro para control de filtro de cluster
              same_cluster: bool = Query(default=True)):
    """
    Genera recomendaciones de canciones similares, con opción de restringir
    la búsqueda al mismo Cluster (tipo de canción) de la canción semilla.
    """
    if artist is None:
        res = resolver_cancion(song_name)
    else:
        res = resolver_cancion_con_artista(song_name, artist)

    if res["status"] in ["Artista_Necesario", "No_Encontrado"]:
        return res

    # single index -> generar recomendaciones
    payload = recomendacion_por_indice(res["index"], top_k=n, same_cluster=same_cluster)
    return {"status": "ok", **payload}


@app.get("/search")
def search(song: str):
    """Busca coincidencias de canciones para sugerir artistas en caso de ambigüedad."""
    name_norm = _norm(song)
    mask = df["song"].astype(str).str.casefold().str.strip() == name_norm
    matches = df.loc[mask, ["song", "artist", "cluster"]].drop_duplicates().to_dict(orient="records")
    
    if not matches:
        return {"status": "No_Encontrado", "Mensaje": "Cancion no encontrada, por favor, pruebe con otro nombre.", "query": song}
    
    artists = sorted({m["artist"] for m in matches if "artist" in m and pd.notna(m["artist"])})
    
    return {"status": "Artista_Necesario" if len(artists) > 1 else "single",
            "query": song,
            "options": {"artists": artists} if len(artists) > 1 else None,
            "matches": matches,
            "count": len(matches)}


@app.get("/")
def root():
    return {"message": "API de Recomendación de Música Híbrido (KNN + K-Means)", "status": "active"}


@app.get("/health")
def health_check():
    return {
        "Grupo": "1",
        "Materia": "Machine Learning",
        "Trabajo Practico": "Recomendador de Música Híbrido (KNN + K-Means)",
        "Estado": "Funcionando",
        "Dataset Cargado": df is not None,
        "Modelo KNN Cargado": knn_model is not None,
        "Escalador Cargado": scaler is not None,
        "Modelo KMeans Cargado": kmeans_model is not None,
        "Modelo Naive Bayes Cargado": naive_bayes_model is not None,
        "Columnas de Características": len(FEATURE_COLS),
        "Clusters": len(nombres_clusters),
        "Filas del Dataset": int(df.shape[0]) if df is not None else 0
    }