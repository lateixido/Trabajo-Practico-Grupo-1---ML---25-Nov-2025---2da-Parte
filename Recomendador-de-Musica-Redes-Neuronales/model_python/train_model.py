import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import euclidean_distances

# Librerías de Deep Learning
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# =============================================================================
# 1. CARGA Y PREPARACIÓN DE DATOS
# =============================================================================
print("--- 1) Cargando datos ---")
# Ajustar ruta si es necesario
df = pd.read_csv(r"C:\Datos\Documentos\Uade\Machine Learning\Trabajo Practico Grupo 1 - ML - 25 Nov 2025 - 2da Parte\dataset\light_spotify_dataset.csv")

id_columnas = ["song", "artist"]
cols_caracteristicas = [
    "Danceability", "Energy", "variance", "Tempo", "Loudness",
    "Acousticness", "Instrumentalness", "Speechiness", 
    "Positiveness", "Popularity", "Liveness"
]

cols_necesarias = id_columnas + cols_caracteristicas

# Validación de columnas
faltante = [c for c in cols_necesarias if c not in df.columns]
if faltante:
    raise ValueError(f"Faltan columnas esperadas: {faltante}")

# Limpieza de nulos
df = df.dropna(subset=cols_necesarias).reset_index(drop=True)

# =============================================================================
# 2. PREPROCESAMIENTO (SCALING)
# =============================================================================
print("--- 2) Escalando features ---")
caracteristicas_numericas = df[cols_caracteristicas].to_numpy(dtype=float)

scaler = StandardScaler()
X_dense = scaler.fit_transform(caracteristicas_numericas)  # Matriz densa para K-Means y Red Neuronal
X_sparse = csr_matrix(X_dense)                # Matriz sparse (opcional, para KNN legacy)

# =============================================================================
# 3. CLUSTERING (K-MEANS) - Contexto auxiliar
# =============================================================================
# Mantener KMeans para tener una etiqueta de 'grupo' general, aunque la recomendación
# final la hará la Red Neuronal.
print("--- 3) Generando Clusters (K-Means) ---")

k_opt = 6  # Ajustar según análisis del codo previo
kmeans = KMeans(n_clusters=k_opt, random_state=42, n_init=10)
df["cluster"] = kmeans.fit_predict(X_dense)

nombres_clusters = {
    0: "Pop Urbano / Rap Melódico / Trap Mainstream",
    1: "Rock/Metal + Rap Intenso + Worship en vivo",
    2: "Vocal Jazz",      # Acousticness
    3: "Rap / Hip Hop",         # Danceability / Speechiness
    4: "Rock/Industrial Atmosférico & Electrónica Oscura",  # Instrumentalness
    5: "Rock/Pop Energético y Optimista"         # Tempo / Positiveness
}


df["nombre_cluster"] = df["cluster"].map(nombres_clusters)

# =============================================================================
# 4. RED NEURONAL (AUTOENCODER)
# =============================================================================
print("\n--- 4) Entrenando Autoencoder ---")

input_dim = X_dense.shape[1]  # Cantidad de features (11 en este caso)
latent_dim = 6                # Dimensión del espacio latente (Embeddings)

# --- Arquitectura del Modelo ---
inputs = Input(shape=(input_dim,))
encoded = Dense(12, activation="relu")(inputs)          # Capa de compresión 1
latent = Dense(latent_dim, activation="relu", name="latent_space")(encoded) # Botella de cuello
decoded = Dense(12, activation="relu")(latent)          # Capa de descompresión 1
outputs = Dense(input_dim, activation="linear")(decoded) # Reconstrucción

autoencoder = Model(inputs, outputs)
encoder = Model(inputs, latent) # Modelo solo para extraer embeddings

autoencoder.compile(optimizer=Adam(1e-3), loss="mse")

# --- Callbacks ---
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

# --- Entrenamiento ---
history = autoencoder.fit(
    X_dense, X_dense,  # Autoencoder: entrada = salida
    epochs=50,
    batch_size=64,
    validation_split=0.1,
    callbacks=[early_stop],
    verbose=1
)

# --- Gráfico de Loss ---
plt.figure(figsize=(8, 4))
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.title("Entrenamiento del Autoencoder")
plt.xlabel("Epochs")
plt.ylabel("MSE")
plt.legend()
plt.show()

# =============================================================================
# 5. GENERACIÓN DE EMBEDDINGS
# =============================================================================
print("--- 5) Generando Embeddings ---")
embeddings = encoder.predict(X_dense)
print(f"Shape de Embeddings: {embeddings.shape}")

# Agregar los embeddings al DataFrame para fácil acceso
embed_cols = [f"embed_{i}" for i in range(latent_dim)]
for i in range(latent_dim):
    df[f"embed_{i}"] = embeddings[:, i]

# =============================================================================
# 6. GUARDADO DE ARTIFACTS
# =============================================================================
print("--- 6) Guardando modelo y artefactos ---")

# Agregar 'cols_caracteristicas' a la lista de columnas a guardar
cols_to_save = ["song", "artist", "cluster", "nombre_cluster"] + cols_caracteristicas + embed_cols

artifacts = {
    "scaler": scaler,
    "cols_caracteristicas": cols_caracteristicas,
    "kmeans_model": kmeans,
    "nombres_clusters": nombres_clusters,
    "autoencoder_model": autoencoder,
    "encoder_model": encoder,
    "embeddings": embeddings,
    # Guardamos el DF con las características (Danceability, Energy, etc.)
    "dataframe_data": df[cols_to_save].copy(), 
    "input_dim": input_dim
}

joblib.dump(artifacts, "music_recommender_neural.joblib")
print("💾 Saved: music_recommender_neural.joblib")

# =============================================================================
# 7. SMOKE TEST (PRUEBA DE FUNCIONAMIENTO)
# =============================================================================
def recomendacion_manual(track_name, artist_name=None, top_k=5):
    # 1. Buscar la canción
    mask = df["song"].str.lower() == track_name.lower()
    if artist_name:
        mask &= df["artist"].str.lower() == artist_name.lower()
    
    matches = df[mask]
    
    if matches.empty:
        return f"Canción '{track_name}' no encontrada."
    
    # Tomar el primer match
    idx = matches.index[0]
    target_vec = matches.loc[idx, embed_cols].values.reshape(1, -1)
    
    # 2. Calcular distancia contra TODOS los embeddings
    # Usar la matriz de embeddings generada anteriormente
    dists = euclidean_distances(target_vec, df[embed_cols].values).flatten()
    
    # 3. Ordenar y devolver
    # argsort devuelve los índices ordenados por distancia menor a mayor
    closest_indices = dists.argsort()[1:top_k+1] # [1:] para saltar la canción misma
    
    print(f"\nRecomendaciones para: {matches.loc[idx, 'song']} - {matches.loc[idx, 'artist']}")
    print(f"Cluster base: {matches.loc[idx, 'nombre_cluster']}")
    print("-" * 50)
    
    for i in closest_indices:
        row = df.iloc[i]
        print(f"* {row['song']} ({row['artist']})")
        print(f"  Cluster: {row['nombre_cluster']} | Distancia: {dists[i]:.4f}")

# Prueba manual
try:
    test_track = input("\nIngresá una canción para probar el modelo neuronal: ").strip()
    if test_track:
        recomendacion_manual(test_track, top_k=5)
except Exception as e:
    print(f"Error en el test: {e}")