"""
Anime Recommender System - Utility Functions
=============================================
Este módulo contiene funciones utilitarias para:
- Carga y preprocesamiento de datos
- Generación de embeddings con Sentence Transformers
- Creación y gestión de índices Faiss
- Cálculo de similitudes y recomendaciones

Autor: Cesar Adrian Delgado
Fecha: Diciembre 2025
"""

import os
import pickle
import logging
from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path

import numpy as np
import pandas as pd
import faiss
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==============================================================================
# CONSTANTES Y CONFIGURACIÓN
# ==============================================================================

DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384  # Dimensión del modelo all-MiniLM-L6-v2
BATCH_SIZE = 32
DATA_DIR = Path("data")
MODELS_DIR = Path("models")
EMBEDDINGS_DIR = Path("embeddings")


# ==============================================================================
# FUNCIONES DE CARGA Y PREPROCESAMIENTO DE DATOS
# ==============================================================================

def load_anime_data(filepath: Union[str, Path]) -> pd.DataFrame:
    """
    Carga el dataset de anime desde un archivo CSV.
    
    Args:
        filepath: Ruta al archivo CSV con datos de anime
        
    Returns:
        DataFrame con los datos de anime procesados
    """
    logger.info(f"Cargando datos desde {filepath}")
    
    df = pd.read_csv(filepath)
    logger.info(f"Dataset cargado: {len(df)} registros, {len(df.columns)} columnas")
    
    return df


def preprocess_anime_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesa el dataset de anime para el sistema de recomendación.
    
    - Limpia valores nulos
    - Normaliza géneros
    - Crea texto combinado para embeddings
    
    Args:
        df: DataFrame con datos de anime crudos
        
    Returns:
        DataFrame preprocesado
    """
    logger.info("Preprocesando datos de anime...")
    
    df_clean = df.copy()
    
    # Columnas esperadas (ajustar según el dataset usado)
    # Nombres comunes: title, name, synopsis, genre, genres, score, rating
    
    # Normalizar nombres de columnas
    column_mapping = {
        'Name': 'title',
        'name': 'title',
        'Title': 'title',
        'Synopsis': 'synopsis',
        'synopsis': 'synopsis',
        'Genre': 'genres',
        'genre': 'genres',
        'Genres': 'genres',
        'genres': 'genres',
        'Score': 'score',
        'score': 'score',
        'Rating': 'rating',
        'rating': 'rating',
        'anime_id': 'anime_id',
        'MAL_ID': 'anime_id',
        'uid': 'anime_id'
    }
    
    df_clean = df_clean.rename(columns={k: v for k, v in column_mapping.items() if k in df_clean.columns})
    
    # Asegurar columna de ID
    if 'anime_id' not in df_clean.columns:
        df_clean['anime_id'] = range(len(df_clean))
    
    # Limpiar sinopsis
    if 'synopsis' in df_clean.columns:
        df_clean['synopsis'] = df_clean['synopsis'].fillna('')
        df_clean['synopsis'] = df_clean['synopsis'].astype(str)
        # Remover textos vacíos o placeholder
        df_clean['synopsis'] = df_clean['synopsis'].replace(['Unknown', 'No synopsis', 'nan'], '')
    else:
        df_clean['synopsis'] = ''
    
    # Limpiar géneros (remover comillas, corchetes, etc.)
    if 'genres' in df_clean.columns:
        df_clean['genres'] = df_clean['genres'].fillna('')
        df_clean['genres'] = df_clean['genres'].astype(str)
        # Aplicar limpieza de géneros
        df_clean['genres'] = df_clean['genres'].apply(clean_genres_string)
    else:
        df_clean['genres'] = ''
    
    # Limpiar título
    if 'title' in df_clean.columns:
        df_clean['title'] = df_clean['title'].fillna('Unknown')
        df_clean['title'] = df_clean['title'].astype(str)
    else:
        df_clean['title'] = 'Unknown'
    
    # Normalizar score/rating
    if 'score' in df_clean.columns:
        df_clean['score'] = pd.to_numeric(df_clean['score'], errors='coerce').fillna(0)
    elif 'rating' in df_clean.columns:
        df_clean['score'] = pd.to_numeric(df_clean['rating'], errors='coerce').fillna(0)
    else:
        df_clean['score'] = 0
    
    # Crear texto combinado para embeddings
    df_clean['combined_text'] = create_combined_text(df_clean)
    
    # Filtrar registros sin información útil
    df_clean = df_clean[df_clean['combined_text'].str.len() > 10].reset_index(drop=True)
    
    logger.info(f"Datos preprocesados: {len(df_clean)} registros válidos")
    
    return df_clean


def create_combined_text(df: pd.DataFrame) -> pd.Series:
    """
    Crea un texto combinado de título, géneros y sinopsis para generar embeddings.
    
    Args:
        df: DataFrame con columnas title, genres, synopsis
        
    Returns:
        Serie con textos combinados
    """
    combined = []
    
    for _, row in df.iterrows():
        parts = []
        
        # Título
        if 'title' in df.columns and pd.notna(row.get('title', '')):
            parts.append(f"Title: {row['title']}")
        
        # Géneros
        if 'genres' in df.columns and pd.notna(row.get('genres', '')):
            genres = str(row['genres'])
            if genres and genres != 'nan':
                parts.append(f"Genres: {genres}")
        
        # Sinopsis
        if 'synopsis' in df.columns and pd.notna(row.get('synopsis', '')):
            synopsis = str(row['synopsis'])
            if synopsis and synopsis != 'nan' and len(synopsis) > 5:
                # Limitar longitud de sinopsis para embeddings
                parts.append(f"Synopsis: {synopsis[:500]}")
        
        combined.append(" | ".join(parts))
    
    return pd.Series(combined)


def clean_genre(genre: str) -> str:
    """
    Limpia un género individual removiendo caracteres no deseados.
    
    Maneja casos como:
    - "'Adventure'" -> "Adventure"
    - "'Adventure']" -> "Adventure"
    - "[Adventure" -> "Adventure"
    - " Adventure " -> "Adventure"
    
    Args:
        genre: String del género a limpiar
        
    Returns:
        Género limpio
    """
    if not genre:
        return ''
    
    # Convertir a string y eliminar espacios
    cleaned = str(genre).strip()
    
    # Remover corchetes, comillas simples, dobles y otros caracteres
    chars_to_remove = ["'", '"', '[', ']', '(', ')', '{', '}']
    for char in chars_to_remove:
        cleaned = cleaned.replace(char, '')
    
    # Limpiar espacios adicionales que puedan quedar
    cleaned = cleaned.strip()
    
    return cleaned


def clean_genres_string(genres_str: str) -> str:
    """
    Limpia una cadena completa de géneros y la devuelve formateada.
    
    Args:
        genres_str: String con géneros (puede tener formato sucio)
        
    Returns:
        String de géneros limpio, separado por comas
    """
    genres = parse_genres(genres_str)
    return ', '.join(genres)


def parse_genres(genres_str: str) -> List[str]:
    """
    Parsea una cadena de géneros separados por coma.
    
    Limpia caracteres no deseados como comillas y corchetes.
    Ejemplo: "['Adventure', 'Action']" -> ["Adventure", "Action"]
    
    Args:
        genres_str: String con géneros separados por coma
        
    Returns:
        Lista de géneros limpios y únicos
    """
    if pd.isna(genres_str) or not genres_str:
        return []
    
    # Convertir a string
    genres_str = str(genres_str)
    
    # Separar por coma y limpiar cada género
    genres = [clean_genre(g) for g in genres_str.split(',')]
    
    # Filtrar vacíos y 'nan'
    genres = [g for g in genres if g and g.lower() != 'nan']
    
    # Eliminar duplicados manteniendo orden
    seen = set()
    unique_genres = []
    for g in genres:
        if g not in seen:
            seen.add(g)
            unique_genres.append(g)
    
    return unique_genres


def get_all_genres(df: pd.DataFrame) -> List[str]:
    """
    Obtiene lista única de todos los géneros en el dataset.
    
    Args:
        df: DataFrame con columna 'genres'
        
    Returns:
        Lista ordenada de géneros únicos
    """
    all_genres = set()
    
    if 'genres' not in df.columns:
        return []
    
    for genres_str in df['genres']:
        genres = parse_genres(genres_str)
        all_genres.update(genres)
    
    return sorted(list(all_genres))


# ==============================================================================
# FUNCIONES DE EMBEDDINGS
# ==============================================================================

def load_embedding_model(model_name: str = DEFAULT_MODEL_NAME) -> SentenceTransformer:
    """
    Carga el modelo de Sentence Transformers para generar embeddings.
    
    Args:
        model_name: Nombre del modelo en HuggingFace Hub
        
    Returns:
        Modelo SentenceTransformer cargado
    """
    logger.info(f"Cargando modelo de embeddings: {model_name}")
    model = SentenceTransformer(model_name)
    logger.info(f"Modelo cargado. Dimensión de embeddings: {model.get_sentence_embedding_dimension()}")
    return model


def generate_embeddings(
    texts: List[str],
    model: SentenceTransformer,
    batch_size: int = BATCH_SIZE,
    show_progress: bool = True
) -> np.ndarray:
    """
    Genera embeddings para una lista de textos.
    
    Args:
        texts: Lista de textos a procesar
        model: Modelo SentenceTransformer
        batch_size: Tamaño del batch para procesamiento
        show_progress: Mostrar barra de progreso
        
    Returns:
        Array numpy con embeddings (n_texts, embedding_dim)
    """
    logger.info(f"Generando embeddings para {len(texts)} textos...")
    
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=True  # Normalizar para búsqueda de similaridad
    )
    
    logger.info(f"Embeddings generados: shape {embeddings.shape}")
    return embeddings


def save_embeddings(embeddings: np.ndarray, filepath: Union[str, Path]) -> None:
    """
    Guarda embeddings en disco.
    
    Args:
        embeddings: Array de embeddings
        filepath: Ruta donde guardar
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    np.save(filepath, embeddings)
    logger.info(f"Embeddings guardados en {filepath}")


def load_embeddings(filepath: Union[str, Path]) -> np.ndarray:
    """
    Carga embeddings desde disco.
    
    Args:
        filepath: Ruta al archivo de embeddings
        
    Returns:
        Array de embeddings
    """
    embeddings = np.load(filepath)
    logger.info(f"Embeddings cargados: shape {embeddings.shape}")
    return embeddings


# ==============================================================================
# FUNCIONES DE FAISS INDEX
# ==============================================================================

def create_faiss_index(
    embeddings: np.ndarray,
    index_type: str = "flat"
) -> faiss.Index:
    """
    Crea un índice Faiss para búsqueda de similaridad.
    
    Args:
        embeddings: Array de embeddings (n, d)
        index_type: Tipo de índice ('flat', 'ivf', 'hnsw')
        
    Returns:
        Índice Faiss configurado
    """
    n_vectors, dimension = embeddings.shape
    logger.info(f"Creando índice Faiss ({index_type}) para {n_vectors} vectores de dim {dimension}")
    
    # Asegurar que los embeddings son float32
    embeddings = embeddings.astype(np.float32)
    
    if index_type == "flat":
        # Índice exacto - mejor para datasets pequeños/medianos
        index = faiss.IndexFlatIP(dimension)  # Inner Product (cosine sim para vectors normalizados)
    
    elif index_type == "ivf":
        # Índice aproximado - mejor para datasets grandes
        n_clusters = min(int(np.sqrt(n_vectors)), 100)
        quantizer = faiss.IndexFlatIP(dimension)
        index = faiss.IndexIVFFlat(quantizer, dimension, n_clusters, faiss.METRIC_INNER_PRODUCT)
        index.train(embeddings)
    
    elif index_type == "hnsw":
        # Hierarchical Navigable Small World - buen balance velocidad/precisión
        index = faiss.IndexHNSWFlat(dimension, 32, faiss.METRIC_INNER_PRODUCT)
    
    else:
        raise ValueError(f"Tipo de índice no soportado: {index_type}")
    
    # Agregar vectores al índice
    index.add(embeddings)
    
    logger.info(f"Índice creado con {index.ntotal} vectores")
    return index


def save_faiss_index(index: faiss.Index, filepath: Union[str, Path]) -> None:
    """
    Guarda índice Faiss en disco.
    
    Args:
        index: Índice Faiss
        filepath: Ruta donde guardar
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    faiss.write_index(index, str(filepath))
    logger.info(f"Índice Faiss guardado en {filepath}")


def load_faiss_index(filepath: Union[str, Path]) -> faiss.Index:
    """
    Carga índice Faiss desde disco.
    
    Args:
        filepath: Ruta al archivo de índice
        
    Returns:
        Índice Faiss
    """
    index = faiss.read_index(str(filepath))
    logger.info(f"Índice Faiss cargado: {index.ntotal} vectores")
    return index


# ==============================================================================
# FUNCIONES DE RECOMENDACIÓN
# ==============================================================================

def search_similar(
    query_embedding: np.ndarray,
    index: faiss.Index,
    k: int = 10
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Busca los k vectores más similares en el índice.
    
    Args:
        query_embedding: Embedding de consulta (1, d) o (d,)
        index: Índice Faiss
        k: Número de resultados
        
    Returns:
        Tupla (distancias, índices)
    """
    # Asegurar forma correcta
    if query_embedding.ndim == 1:
        query_embedding = query_embedding.reshape(1, -1)
    
    query_embedding = query_embedding.astype(np.float32)
    
    distances, indices = index.search(query_embedding, k)
    
    return distances[0], indices[0]


def get_recommendations_by_text(
    query_text: str,
    model: SentenceTransformer,
    index: faiss.Index,
    df: pd.DataFrame,
    k: int = 10,
    min_score: float = 0.0
) -> pd.DataFrame:
    """
    Obtiene recomendaciones basadas en un texto de búsqueda.
    
    Args:
        query_text: Texto de búsqueda (descripción de anime deseado)
        model: Modelo de embeddings
        index: Índice Faiss
        df: DataFrame con datos de anime
        k: Número de recomendaciones
        min_score: Score mínimo del anime (filtro de calidad)
        
    Returns:
        DataFrame con recomendaciones ordenadas por similaridad
    """
    # Generar embedding de la consulta
    query_embedding = model.encode([query_text], normalize_embeddings=True)
    
    # Buscar similares
    similarities, indices = search_similar(query_embedding, index, k * 2)  # Extra por filtros
    
    # Construir resultados
    results = []
    for sim, idx in zip(similarities, indices):
        if idx < len(df):
            anime = df.iloc[idx].to_dict()
            anime['similarity_score'] = float(sim)
            results.append(anime)
    
    results_df = pd.DataFrame(results)
    
    # Filtrar por score mínimo si aplica
    if min_score > 0 and 'score' in results_df.columns:
        results_df = results_df[results_df['score'] >= min_score]
    
    # Retornar top k
    return results_df.head(k)


def get_recommendations_by_anime(
    anime_id: int,
    index: faiss.Index,
    df: pd.DataFrame,
    embeddings: np.ndarray,
    k: int = 10,
    exclude_self: bool = True
) -> pd.DataFrame:
    """
    Obtiene recomendaciones similares a un anime específico.
    
    Args:
        anime_id: ID del anime (índice en el DataFrame)
        index: Índice Faiss
        df: DataFrame con datos de anime
        embeddings: Array de embeddings
        k: Número de recomendaciones
        exclude_self: Excluir el anime original de resultados
        
    Returns:
        DataFrame con recomendaciones
    """
    # Obtener embedding del anime
    query_embedding = embeddings[anime_id:anime_id+1]
    
    # Buscar similares
    k_search = k + 1 if exclude_self else k
    similarities, indices = search_similar(query_embedding, index, k_search)
    
    # Construir resultados
    results = []
    for sim, idx in zip(similarities, indices):
        if exclude_self and idx == anime_id:
            continue
        if idx < len(df):
            anime = df.iloc[idx].to_dict()
            anime['similarity_score'] = float(sim)
            results.append(anime)
    
    return pd.DataFrame(results).head(k)


def get_recommendations_hybrid(
    query_text: str,
    model: SentenceTransformer,
    index: faiss.Index,
    df: pd.DataFrame,
    k: int = 10,
    genres_filter: Optional[List[str]] = None,
    min_score: float = 0.0,
    score_weight: float = 0.2
) -> pd.DataFrame:
    """
    Sistema de recomendación híbrido que combina:
    - Similaridad semántica (embeddings)
    - Filtro por géneros
    - Ponderación por rating
    
    Args:
        query_text: Texto de búsqueda
        model: Modelo de embeddings
        index: Índice Faiss
        df: DataFrame con datos
        k: Número de recomendaciones
        genres_filter: Lista de géneros requeridos (opcional)
        min_score: Score mínimo
        score_weight: Peso del rating en score final (0-1)
        
    Returns:
        DataFrame con recomendaciones híbridas
    """
    # Obtener más candidatos para filtrar
    candidates = get_recommendations_by_text(
        query_text, model, index, df, k=k*3, min_score=0
    )
    
    # Filtrar por géneros si se especificaron
    if genres_filter:
        def has_genres(genres_str):
            anime_genres = set(parse_genres(genres_str))
            return bool(anime_genres.intersection(set(genres_filter)))
        
        candidates = candidates[candidates['genres'].apply(has_genres)]
    
    # Filtrar por score mínimo
    if min_score > 0 and 'score' in candidates.columns:
        candidates = candidates[candidates['score'] >= min_score]
    
    # Calcular score híbrido
    if 'score' in candidates.columns and len(candidates) > 0:
        # Normalizar scores a 0-1
        max_rating = candidates['score'].max()
        min_rating = candidates['score'].min()
        rating_range = max_rating - min_rating if max_rating > min_rating else 1
        
        candidates['normalized_rating'] = (candidates['score'] - min_rating) / rating_range
        
        # Score híbrido: combinar similaridad semántica con rating
        candidates['hybrid_score'] = (
            (1 - score_weight) * candidates['similarity_score'] +
            score_weight * candidates['normalized_rating']
        )
        
        # Ordenar por score híbrido
        candidates = candidates.sort_values('hybrid_score', ascending=False)
    
    return candidates.head(k)


# ==============================================================================
# FUNCIONES DE UTILIDAD GENERAL
# ==============================================================================

def format_recommendation(anime: Dict, rank: int = 0) -> str:
    """
    Formatea un anime recomendado para mostrar.
    
    Args:
        anime: Diccionario con datos del anime
        rank: Posición en el ranking
        
    Returns:
        String formateado
    """
    title = anime.get('title', 'Unknown')
    genres = anime.get('genres', 'N/A')
    score = anime.get('score', 0)
    sim = anime.get('similarity_score', 0)
    
    output = []
    if rank > 0:
        output.append(f"#{rank} 📺 {title}")
    else:
        output.append(f"📺 {title}")
    output.append(f"   Géneros: {genres}")
    output.append(f"   Rating: {score:.1f}/10")
    output.append(f"   Similaridad: {sim:.3f}")
    
    return "\n".join(output)


def print_recommendations(df: pd.DataFrame, title: str = "Recomendaciones") -> None:
    """
    Imprime recomendaciones de forma formateada.
    
    Args:
        df: DataFrame con recomendaciones
        title: Título a mostrar
    """
    print(f"\n{'='*60}")
    print(f"🎬 {title}")
    print(f"{'='*60}\n")
    
    for i, (_, anime) in enumerate(df.iterrows(), 1):
        print(format_recommendation(anime.to_dict(), rank=i))
        print()


def create_sample_data() -> pd.DataFrame:
    """
    Crea datos de ejemplo para pruebas cuando no hay dataset disponible.
    
    Returns:
        DataFrame con datos de anime de ejemplo
    """
    sample_data = [
        {
            'anime_id': 1,
            'title': 'Attack on Titan',
            'genres': 'Action, Drama, Fantasy, Mystery',
            'synopsis': 'Centuries ago, mankind was slaughtered to near extinction by monstrous humanoid creatures called Titans. Eren Yeager joins the military to fight back.',
            'score': 9.0
        },
        {
            'anime_id': 2,
            'title': 'Death Note',
            'genres': 'Mystery, Psychological, Supernatural, Thriller',
            'synopsis': 'A high school student discovers a supernatural notebook that allows him to kill anyone by writing their name in it.',
            'score': 9.0
        },
        {
            'anime_id': 3,
            'title': 'Fullmetal Alchemist: Brotherhood',
            'genres': 'Action, Adventure, Drama, Fantasy',
            'synopsis': 'Two brothers search for the Philosophers Stone to restore their bodies after a failed alchemical experiment.',
            'score': 9.1
        },
        {
            'anime_id': 4,
            'title': 'Steins;Gate',
            'genres': 'Drama, Sci-Fi, Thriller',
            'synopsis': 'A self-proclaimed mad scientist accidentally creates a time machine and must deal with the consequences.',
            'score': 9.1
        },
        {
            'anime_id': 5,
            'title': 'Your Name',
            'genres': 'Drama, Romance, Supernatural',
            'synopsis': 'Two teenagers share a connection when they switch bodies, leading to a quest to find each other.',
            'score': 8.9
        },
        {
            'anime_id': 6,
            'title': 'Demon Slayer',
            'genres': 'Action, Fantasy, Shounen',
            'synopsis': 'A boy becomes a demon slayer to avenge his family and cure his sister who has been turned into a demon.',
            'score': 8.5
        },
        {
            'anime_id': 7,
            'title': 'My Hero Academia',
            'genres': 'Action, Comedy, Shounen',
            'synopsis': 'In a world where most people have superpowers, a powerless boy dreams of becoming a hero.',
            'score': 8.0
        },
        {
            'anime_id': 8,
            'title': 'One Piece',
            'genres': 'Action, Adventure, Comedy, Fantasy',
            'synopsis': 'A young pirate searches for the ultimate treasure known as One Piece to become the Pirate King.',
            'score': 8.7
        },
        {
            'anime_id': 9,
            'title': 'Spirited Away',
            'genres': 'Adventure, Drama, Fantasy, Supernatural',
            'synopsis': 'A young girl enters a magical world and must work in a bathhouse for spirits to rescue her parents.',
            'score': 8.8
        },
        {
            'anime_id': 10,
            'title': 'Cowboy Bebop',
            'genres': 'Action, Adventure, Drama, Sci-Fi',
            'synopsis': 'A ragtag crew of bounty hunters chase criminals across the solar system.',
            'score': 8.8
        }
    ]
    
    return pd.DataFrame(sample_data)


# ==============================================================================
# CLASE PRINCIPAL DEL SISTEMA DE RECOMENDACIÓN
# ==============================================================================

class AnimeRecommender:
    """
    Sistema de recomendación de anime basado en embeddings.
    
    Combina:
    - Embeddings de texto (Sentence Transformers)
    - Búsqueda vectorial (Faiss)
    - Filtros por géneros y rating
    """
    
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        index_type: str = "flat"
    ):
        """
        Inicializa el sistema de recomendación.
        
        Args:
            model_name: Nombre del modelo de embeddings
            index_type: Tipo de índice Faiss
        """
        self.model_name = model_name
        self.index_type = index_type
        
        self.model: Optional[SentenceTransformer] = None
        self.index: Optional[faiss.Index] = None
        self.df: Optional[pd.DataFrame] = None
        self.embeddings: Optional[np.ndarray] = None
        
        self._is_initialized = False
    
    def load_model(self) -> None:
        """Carga el modelo de embeddings."""
        self.model = load_embedding_model(self.model_name)
    
    def load_data(self, filepath: Union[str, Path]) -> None:
        """
        Carga y preprocesa datos de anime.
        
        Args:
            filepath: Ruta al archivo CSV
        """
        df_raw = load_anime_data(filepath)
        self.df = preprocess_anime_data(df_raw)
    
    def use_sample_data(self) -> None:
        """Usa datos de ejemplo para pruebas."""
        df_raw = create_sample_data()
        self.df = preprocess_anime_data(df_raw)
        logger.info("Usando datos de ejemplo")
    
    def build_index(self) -> None:
        """Genera embeddings y construye el índice Faiss."""
        if self.model is None:
            self.load_model()
        
        if self.df is None:
            raise ValueError("No hay datos cargados. Use load_data() primero.")
        
        # Generar embeddings
        texts = self.df['combined_text'].tolist()
        self.embeddings = generate_embeddings(texts, self.model)
        
        # Crear índice
        self.index = create_faiss_index(self.embeddings, self.index_type)
        
        self._is_initialized = True
        logger.info("Sistema de recomendación inicializado correctamente")
    
    def save(self, directory: Union[str, Path]) -> None:
        """
        Guarda el estado del recomendador (embeddings, índice, datos).
        
        Args:
            directory: Directorio donde guardar
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        
        # Guardar embeddings
        if self.embeddings is not None:
            save_embeddings(self.embeddings, directory / "embeddings.npy")
        
        # Guardar índice
        if self.index is not None:
            save_faiss_index(self.index, directory / "faiss.index")
        
        # Guardar DataFrame
        if self.df is not None:
            self.df.to_pickle(directory / "anime_data.pkl")
        
        # Guardar configuración
        config = {
            'model_name': self.model_name,
            'index_type': self.index_type
        }
        with open(directory / "config.pkl", 'wb') as f:
            pickle.dump(config, f)
        
        logger.info(f"Recomendador guardado en {directory}")
    
    def load(self, directory: Union[str, Path]) -> None:
        """
        Carga el estado del recomendador desde disco.
        
        Args:
            directory: Directorio con archivos guardados
        """
        directory = Path(directory)
        
        # Cargar configuración
        with open(directory / "config.pkl", 'rb') as f:
            config = pickle.load(f)
        self.model_name = config['model_name']
        self.index_type = config['index_type']
        
        # Cargar modelo
        self.load_model()
        
        # Cargar embeddings
        self.embeddings = load_embeddings(directory / "embeddings.npy")
        
        # Cargar índice
        self.index = load_faiss_index(directory / "faiss.index")
        
        # Cargar DataFrame
        self.df = pd.read_pickle(directory / "anime_data.pkl")
        
        self._is_initialized = True
        logger.info(f"Recomendador cargado desde {directory}")
    
    def recommend_by_text(
        self,
        query: str,
        k: int = 10,
        genres: Optional[List[str]] = None,
        min_score: float = 0.0,
        hybrid: bool = True
    ) -> pd.DataFrame:
        """
        Obtiene recomendaciones basadas en descripción de texto.
        
        Args:
            query: Descripción del anime deseado
            k: Número de recomendaciones
            genres: Filtro de géneros (opcional)
            min_score: Score mínimo
            hybrid: Usar scoring híbrido (incluye rating)
            
        Returns:
            DataFrame con recomendaciones
        """
        if not self._is_initialized:
            raise RuntimeError("Recomendador no inicializado. Use build_index() o load() primero.")
        
        if hybrid:
            return get_recommendations_hybrid(
                query, self.model, self.index, self.df,
                k=k, genres_filter=genres, min_score=min_score
            )
        else:
            return get_recommendations_by_text(
                query, self.model, self.index, self.df,
                k=k, min_score=min_score
            )
    
    def recommend_similar(
        self,
        anime_id: int,
        k: int = 10
    ) -> pd.DataFrame:
        """
        Obtiene animes similares a uno dado.
        
        Args:
            anime_id: ID del anime
            k: Número de recomendaciones
            
        Returns:
            DataFrame con animes similares
        """
        if not self._is_initialized:
            raise RuntimeError("Recomendador no inicializado. Use build_index() o load() primero.")
        
        return get_recommendations_by_anime(
            anime_id, self.index, self.df, self.embeddings, k=k
        )
    
    def search_anime(self, title: str) -> pd.DataFrame:
        """
        Busca anime por título.
        
        Args:
            title: Título a buscar
            
        Returns:
            DataFrame con resultados
        """
        if self.df is None:
            raise RuntimeError("No hay datos cargados.")
        
        mask = self.df['title'].str.lower().str.contains(title.lower(), na=False)
        return self.df[mask]
    
    def get_genres(self) -> List[str]:
        """Obtiene lista de géneros disponibles."""
        if self.df is None:
            return []
        return get_all_genres(self.df)
    
    def get_anime_by_id(self, anime_id: int) -> Optional[Dict]:
        """Obtiene datos de un anime por su ID."""
        if self.df is None:
            return None
        
        matches = self.df[self.df['anime_id'] == anime_id]
        if len(matches) == 0:
            # Intentar por índice
            if anime_id < len(self.df):
                return self.df.iloc[anime_id].to_dict()
            return None
        
        return matches.iloc[0].to_dict()


# ==============================================================================
# EJEMPLO DE USO
# ==============================================================================

if __name__ == "__main__":
    # Demostración con datos de ejemplo
    print("=" * 60)
    print("Anime Recommender System - Demo")
    print("=" * 60)
    
    # Crear instancia
    recommender = AnimeRecommender()
    
    # Usar datos de ejemplo
    recommender.use_sample_data()
    
    # Construir índice
    recommender.build_index()
    
    # Probar recomendaciones
    print("\n--- Recomendaciones por texto ---")
    query = "I want an action anime with supernatural elements and mystery"
    results = recommender.recommend_by_text(query, k=5)
    print_recommendations(results, f"Búsqueda: '{query}'")
    
    print("\n--- Animes similares ---")
    similar = recommender.recommend_similar(0, k=5)  # Similar a Attack on Titan
    print_recommendations(similar, "Similares a Attack on Titan")
