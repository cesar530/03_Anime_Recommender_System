# 🎬 Anime Recommender System

Sistema de recomendación de anime basado en **embeddings semánticos** que combina NLP moderno con búsqueda vectorial eficiente.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com)
[![HuggingFace](https://img.shields.io/badge/🤗-Transformers-yellow.svg)](https://huggingface.co)
[![Faiss](https://img.shields.io/badge/Faiss-Vector_Search-orange.svg)](https://github.com/facebookresearch/faiss)

---

## 📋 Descripción

Este proyecto implementa un sistema de recomendación de anime que utiliza técnicas avanzadas de NLP y ML:

- **Embeddings de texto** con Sentence Transformers (BERT)
- **Búsqueda vectorial** con Faiss para similaridad semántica en milisegundos
- **Scoring híbrido** que combina similaridad semántica con ratings
- **API REST** con FastAPI para servir recomendaciones en producción

### 🎯 Casos de Uso

1. **Búsqueda por descripción**: "Quiero un anime de acción con poderes sobrenaturales"
2. **Anime similar**: "Si te gustó Death Note, te gustará..."
3. **Filtros avanzados**: Combina géneros + rating mínimo + búsqueda semántica

---

## 🏗️ Arquitectura

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   User Query    │────▶│ Sentence-BERT    │────▶│   Query         │
│   "action..."   │     │ Embedding Model  │     │   Embedding     │
└─────────────────┘     └──────────────────┘     └────────┬────────┘
                                                          │
                                                          ▼
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│ Recommendations │◀────│     Ranking      │◀────│   Faiss Index   │
│   (Top K)       │     │ (Hybrid Score)   │     │   (Similarity)  │
└─────────────────┘     └──────────────────┘     └─────────────────┘
```

---

## 🚀 Quick Start

### 1. Clonar el repositorio

```bash
git clone https://github.com/tu-usuario/anime-recommender-system.git
cd anime-recommender-system
```

### 2. Crear entorno virtual

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4. Ejecutar el Notebook

```bash
jupyter notebook anime_recommender_notebook.ipynb
```

### 5. Ejecutar la API

```bash
uvicorn anime_recommender:app --reload --host 0.0.0.0 --port 8000
```

La documentación interactiva estará disponible en:

- **Swagger UI**: <http://localhost:8000/docs>
- **ReDoc**: <http://localhost:8000/redoc>

---

## 📁 Estructura del Proyecto

```
03_Anime_Recommender_System/
│
├── 📓 anime_recommender_notebook.ipynb  # Notebook principal con pipeline completo
├── 🐍 anime_recommender.py              # API FastAPI
├── 🔧 utils.py                          # Funciones utilitarias y clase principal
├── 📋 requirements.txt                  # Dependencias Python
├── 🚫 .gitignore                        # Archivos ignorados por Git
├── 📖 README.md                         # Este archivo
│
├── 📂 data/                             # Datasets (no incluido en Git)
│   └── anime.csv
│
├── 📂 models/                           # Modelos guardados
│   └── anime_recommender/
│       ├── embeddings.npy
│       ├── faiss.index
│       ├── anime_data.pkl
│       └── config.pkl
│
└── 📂 embeddings/                       # Cache de embeddings
```

---

## 🛠️ Tecnologías

| Categoría | Tecnología | Uso |
| --------- | ---------- | --- |
| **NLP** | Sentence Transformers | Generación de embeddings semánticos |
| **ML** | HuggingFace Transformers | Modelos pre-entrenados BERT |
| **Vector Search** | Faiss | Búsqueda de similaridad eficiente |
| **API** | FastAPI | Servidor REST de alta performance |
| **Data** | Pandas, NumPy | Procesamiento de datos |
| **Viz** | Matplotlib, Seaborn | Visualización y análisis |

---

## 📊 Dataset

El sistema puede trabajar con diferentes datasets de anime. Recomendados:

1. **[MyAnimeList Dataset](https://www.kaggle.com/datasets/azathoth42/myanimelist)** - 14,000+ anime con metadata completa
2. **[Anime Recommendations Database](https://www.kaggle.com/datasets/CooperUnion/anime-recommendations-database)** - 12,000+ anime con ratings
3. **[Anime Dataset with Reviews](https://www.kaggle.com/datasets/marlesson/myanimelist-dataset-animes-profiles-reviews)** - Incluye reviews y sinopsis

### Estructura esperada del CSV

| Columna | Descripción |
| ------- | ----------- |
| `title` / `name` | Nombre del anime |
| `synopsis` | Descripción/sinopsis |
| `genres` / `genre` | Géneros separados por coma |
| `score` / `rating` | Rating promedio (0-10) |

---

## 🔌 API Endpoints

### Recomendaciones

```http
POST /recommend/text
Content-Type: application/json

{
  "query": "I want an action anime with supernatural powers",
  "k": 10,
  "genres": ["Action", "Fantasy"],
  "min_score": 7.5,
  "use_hybrid": true
}
```

### Anime Similar

```http
GET /recommend/similar/{anime_id}?k=10
```

### Buscar Anime

```http
GET /anime/search?q=death+note&limit=5
```

### Géneros Disponibles

```http
GET /genres
```

---

## 💡 Características Técnicas

### Embeddings Semánticos

- **Modelo**: `sentence-transformers/all-MiniLM-L6-v2`
- **Dimensión**: 384
- **Normalización**: L2 para similaridad coseno

### Índice Faiss

- **Tipo**: IndexFlatIP (Inner Product)
- **Métrica**: Similaridad coseno (vectores normalizados)
- **Escalabilidad**: Soporta millones de vectores

### Scoring Híbrido

```python
hybrid_score = (1 - α) × similarity_score + α × normalized_rating
```

Donde `α = 0.2` por defecto (20% rating, 80% similaridad semántica).

---

## 📈 Mejoras Futuras

- [ ] Integrar collaborative filtering (usuarios similares)
- [ ] Fine-tuning del modelo BERT con datos de anime
- [ ] Caché de queries frecuentes con Redis
- [ ] Frontend con React/Vue
- [ ] Deploy en cloud (AWS/GCP/Azure)
- [ ] A/B testing de diferentes modelos

---

## 🧪 Pruebas

```bash
# Ejecutar prueba básica del sistema
python utils.py

# Verificar API
curl http://localhost:8000/health
```

---

## 📚 Referencias

- [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://arxiv.org/abs/1908.10084)
- [Faiss: A Library for Efficient Similarity Search](https://github.com/facebookresearch/faiss)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers/)

---

## 👤 Autor

- 👤 Autor : **César Adrián Delgado Díaz**
- 💼 LinkedIn: [linkedin.com/in/cesar-delgado-diaz](linkedin.com/in/cesar-delgado-diaz)
- 🐙 GitHub: [github.com/tu-usuario](https://github.com/cesar530)

---

## 🙏 Agradecimientos

- MyAnimeList por los datos
- HuggingFace por los modelos pre-entrenados
- Facebook Research por Faiss
- La comunidad de anime y desarrolladores

---
