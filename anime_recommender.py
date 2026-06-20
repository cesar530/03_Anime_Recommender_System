"""
Anime Recommender System - FastAPI Application
==============================================
API REST para servir recomendaciones de anime basadas en embeddings semánticos.

Endpoints:
- GET /: Información de la API
- GET /health: Health check
- POST /recommend/text: Recomendaciones por descripción
- POST /recommend/similar/{anime_id}: Animes similares
- GET /anime/search: Buscar anime por título
- GET /anime/{anime_id}: Obtener detalles de un anime
- GET /genres: Lista de géneros disponibles

Para ejecutar:
    uvicorn anime_recommender:app --reload --host 0.0.0.0 --port 8000

Autor: [Tu Nombre]
Fecha: Diciembre 2024
"""

import os
import logging
from pathlib import Path
from typing import List, Optional
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from utils import AnimeRecommender, create_sample_data

# ==============================================================================
# CONFIGURACIÓN
# ==============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Rutas de archivos
PROJECT_DIR = Path(__file__).parent
MODELS_DIR = PROJECT_DIR / "models" / "anime_recommender"

# Instancia global del recomendador
recommender: Optional[AnimeRecommender] = None


# ==============================================================================
# MODELOS PYDANTIC (Esquemas de Request/Response)
# ==============================================================================

class TextRecommendationRequest(BaseModel):
    """Request para recomendación por texto."""
    query: str = Field(
        ...,
        description="Descripción del tipo de anime que buscas",
        min_length=3,
        max_length=500,
        example="I want an action anime with supernatural powers and intense battles"
    )
    k: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Número de recomendaciones"
    )
    genres: Optional[List[str]] = Field(
        default=None,
        description="Filtrar por géneros específicos",
        example=["Action", "Fantasy"]
    )
    min_score: float = Field(
        default=0.0,
        ge=0,
        le=10,
        description="Score mínimo del anime"
    )
    use_hybrid: bool = Field(
        default=True,
        description="Usar scoring híbrido (combina similaridad + rating)"
    )


class AnimeRecommendation(BaseModel):
    """Modelo de una recomendación de anime."""
    anime_id: int
    title: str
    genres: str
    synopsis: Optional[str] = None
    score: float
    similarity_score: float
    hybrid_score: Optional[float] = None


class RecommendationResponse(BaseModel):
    """Response con lista de recomendaciones."""
    query: str
    total_results: int
    recommendations: List[AnimeRecommendation]


class AnimeDetail(BaseModel):
    """Detalle completo de un anime."""
    anime_id: int
    title: str
    genres: str
    synopsis: str
    score: float


class GenresResponse(BaseModel):
    """Response con lista de géneros."""
    total: int
    genres: List[str]


class HealthResponse(BaseModel):
    """Response del health check."""
    status: str
    model_loaded: bool
    total_anime: int
    index_size: int


class APIInfo(BaseModel):
    """Información de la API."""
    name: str
    version: str
    description: str
    endpoints: List[dict]


# ==============================================================================
# INICIALIZACIÓN DE LA APLICACIÓN
# ==============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifecycle manager para cargar el modelo al inicio.
    """
    global recommender
    
    logger.info("🚀 Iniciando Anime Recommender API...")
    
    # Intentar cargar modelo guardado o crear uno nuevo
    recommender = AnimeRecommender()
    
    if MODELS_DIR.exists():
        try:
            logger.info(f"📂 Cargando modelo desde {MODELS_DIR}")
            recommender.load(MODELS_DIR)
            logger.info(f"✅ Modelo cargado: {len(recommender.df)} anime indexados")
        except Exception as e:
            logger.warning(f"⚠️ No se pudo cargar modelo: {e}")
            logger.info("📦 Inicializando con datos de ejemplo...")
            _initialize_with_sample_data()
    else:
        logger.info("📦 No hay modelo guardado. Inicializando con datos de ejemplo...")
        _initialize_with_sample_data()
    
    yield  # La aplicación está corriendo
    
    # Cleanup al cerrar
    logger.info("👋 Cerrando Anime Recommender API...")


def _initialize_with_sample_data():
    """Inicializa el recomendador con datos de ejemplo."""
    global recommender
    recommender.use_sample_data()
    recommender.build_index()
    
    # Guardar para uso futuro
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    recommender.save(MODELS_DIR)
    logger.info(f"✅ Modelo inicializado y guardado en {MODELS_DIR}")


# Crear aplicación FastAPI
app = FastAPI(
    title="🎬 Anime Recommender API",
    description="""
## Sistema de Recomendación de Anime basado en Embeddings Semánticos

Esta API proporciona recomendaciones de anime usando:
- **Sentence Transformers** para embeddings de texto
- **Faiss** para búsqueda vectorial eficiente
- **Scoring híbrido** combinando similaridad semántica y ratings

### Características:
- 🔍 Búsqueda semántica por descripción
- 🎯 Recomendaciones de anime similar
- 🎭 Filtros por género y rating
- ⚡ Respuestas en milisegundos

### Autor
Proyecto de portafolio para demostrar skills en ML/NLP y sistemas de recomendación.
    """,
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configurar CORS para permitir requests desde cualquier origen
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==============================================================================
# ENDPOINTS
# ==============================================================================

@app.get("/", response_model=APIInfo, tags=["General"])
async def root():
    """
    Información general de la API.
    """
    return APIInfo(
        name="Anime Recommender API",
        version="1.0.0",
        description="Sistema de recomendación de anime basado en embeddings semánticos",
        endpoints=[
            {"method": "GET", "path": "/health", "description": "Health check"},
            {"method": "POST", "path": "/recommend/text", "description": "Recomendaciones por descripción"},
            {"method": "GET", "path": "/recommend/similar/{anime_id}", "description": "Anime similar"},
            {"method": "GET", "path": "/anime/search", "description": "Buscar anime"},
            {"method": "GET", "path": "/anime/{anime_id}", "description": "Detalles de anime"},
            {"method": "GET", "path": "/genres", "description": "Lista de géneros"},
        ]
    )


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """
    Verifica el estado de la API y el modelo.
    """
    global recommender
    
    is_loaded = recommender is not None and recommender._is_initialized
    total_anime = len(recommender.df) if is_loaded else 0
    index_size = recommender.index.ntotal if is_loaded and recommender.index else 0
    
    return HealthResponse(
        status="healthy" if is_loaded else "initializing",
        model_loaded=is_loaded,
        total_anime=total_anime,
        index_size=index_size
    )


@app.post("/recommend/text", response_model=RecommendationResponse, tags=["Recomendaciones"])
async def recommend_by_text(request: TextRecommendationRequest):
    """
    Obtiene recomendaciones basadas en una descripción textual.
    
    Describe el tipo de anime que buscas y el sistema encontrará los más similares
    usando búsqueda semántica con embeddings.
    
    **Ejemplos de queries:**
    - "I want an action anime with supernatural powers"
    - "A romantic comedy set in high school"
    - "Dark psychological thriller with plot twists"
    """
    global recommender
    
    if recommender is None or not recommender._is_initialized:
        raise HTTPException(status_code=503, detail="Modelo no inicializado")
    
    try:
        # Obtener recomendaciones
        results = recommender.recommend_by_text(
            query=request.query,
            k=request.k,
            genres=request.genres,
            min_score=request.min_score,
            hybrid=request.use_hybrid
        )
        
        # Convertir a formato de respuesta
        recommendations = []
        for _, row in results.iterrows():
            rec = AnimeRecommendation(
                anime_id=int(row.get('anime_id', 0)),
                title=str(row.get('title', 'Unknown')),
                genres=str(row.get('genres', '')),
                synopsis=str(row.get('synopsis', ''))[:300] if row.get('synopsis') else None,
                score=float(row.get('score', 0)),
                similarity_score=float(row.get('similarity_score', 0)),
                hybrid_score=float(row.get('hybrid_score', 0)) if 'hybrid_score' in row else None
            )
            recommendations.append(rec)
        
        return RecommendationResponse(
            query=request.query,
            total_results=len(recommendations),
            recommendations=recommendations
        )
    
    except Exception as e:
        logger.error(f"Error en recomendación: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/recommend/similar/{anime_id}", response_model=RecommendationResponse, tags=["Recomendaciones"])
async def recommend_similar(
    anime_id: int,
    k: int = Query(default=10, ge=1, le=50, description="Número de recomendaciones")
):
    """
    Obtiene anime similares a uno específico.
    
    Útil para: "Si te gustó X, te gustará..."
    """
    global recommender
    
    if recommender is None or not recommender._is_initialized:
        raise HTTPException(status_code=503, detail="Modelo no inicializado")
    
    # Verificar que el anime existe
    anime = recommender.get_anime_by_id(anime_id)
    if anime is None:
        raise HTTPException(status_code=404, detail=f"Anime con ID {anime_id} no encontrado")
    
    try:
        results = recommender.recommend_similar(anime_id, k=k)
        
        recommendations = []
        for _, row in results.iterrows():
            rec = AnimeRecommendation(
                anime_id=int(row.get('anime_id', 0)),
                title=str(row.get('title', 'Unknown')),
                genres=str(row.get('genres', '')),
                synopsis=str(row.get('synopsis', ''))[:300] if row.get('synopsis') else None,
                score=float(row.get('score', 0)),
                similarity_score=float(row.get('similarity_score', 0))
            )
            recommendations.append(rec)
        
        return RecommendationResponse(
            query=f"Similar to: {anime['title']}",
            total_results=len(recommendations),
            recommendations=recommendations
        )
    
    except Exception as e:
        logger.error(f"Error en recomendación similar: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/anime/search", tags=["Anime"])
async def search_anime(
    q: str = Query(..., min_length=1, description="Término de búsqueda"),
    limit: int = Query(default=10, ge=1, le=50, description="Máximo de resultados")
):
    """
    Busca anime por título.
    """
    global recommender
    
    if recommender is None or not recommender._is_initialized:
        raise HTTPException(status_code=503, detail="Modelo no inicializado")
    
    results = recommender.search_anime(q)
    
    anime_list = []
    for _, row in results.head(limit).iterrows():
        anime_list.append({
            "anime_id": int(row.get('anime_id', 0)),
            "title": str(row.get('title', 'Unknown')),
            "genres": str(row.get('genres', '')),
            "score": float(row.get('score', 0))
        })
    
    return {
        "query": q,
        "total_results": len(anime_list),
        "results": anime_list
    }


@app.get("/anime/{anime_id}", response_model=AnimeDetail, tags=["Anime"])
async def get_anime(anime_id: int):
    """
    Obtiene detalles de un anime específico por su ID.
    """
    global recommender
    
    if recommender is None or not recommender._is_initialized:
        raise HTTPException(status_code=503, detail="Modelo no inicializado")
    
    anime = recommender.get_anime_by_id(anime_id)
    
    if anime is None:
        raise HTTPException(status_code=404, detail=f"Anime con ID {anime_id} no encontrado")
    
    return AnimeDetail(
        anime_id=int(anime.get('anime_id', anime_id)),
        title=str(anime.get('title', 'Unknown')),
        genres=str(anime.get('genres', '')),
        synopsis=str(anime.get('synopsis', '')),
        score=float(anime.get('score', 0))
    )


@app.get("/genres", response_model=GenresResponse, tags=["Metadata"])
async def get_genres():
    """
    Obtiene la lista de todos los géneros disponibles.
    
    Útil para construir filtros en la UI.
    """
    global recommender
    
    if recommender is None or not recommender._is_initialized:
        raise HTTPException(status_code=503, detail="Modelo no inicializado")
    
    genres = recommender.get_genres()
    
    return GenresResponse(
        total=len(genres),
        genres=genres
    )


@app.get("/stats", tags=["Metadata"])
async def get_stats():
    """
    Obtiene estadísticas del dataset.
    """
    global recommender
    
    if recommender is None or not recommender._is_initialized:
        raise HTTPException(status_code=503, detail="Modelo no inicializado")
    
    df = recommender.df
    genres = recommender.get_genres()
    
    return {
        "total_anime": len(df),
        "total_genres": len(genres),
        "score_stats": {
            "mean": float(df['score'].mean()) if 'score' in df.columns else 0,
            "min": float(df['score'].min()) if 'score' in df.columns else 0,
            "max": float(df['score'].max()) if 'score' in df.columns else 0
        },
        "top_genres": genres[:10],
        "embedding_dimension": recommender.embeddings.shape[1] if recommender.embeddings is not None else 0,
        "index_vectors": recommender.index.ntotal if recommender.index else 0
    }


# ==============================================================================
# PUNTO DE ENTRADA
# ==============================================================================

if __name__ == "__main__":
    import uvicorn
    
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║           🎬 Anime Recommender API                       ║
    ║                                                          ║
    ║   Documentación: http://localhost:8000/docs              ║
    ║   ReDoc:         http://localhost:8000/redoc             ║
    ║   Health:        http://localhost:8000/health            ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    uvicorn.run(
        "anime_recommender:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
