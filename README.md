# Sri Lankan Legal AI Data Pipeline

A comprehensive data pipeline for building a legal research AI system using Sri Lankan Supreme Court documents. This system downloads, processes, and indexes legal judgments to enable semantic search and fine-tunes embedding models for improved retrieval.

## 📋 Pipeline Overview

The pipeline consists of 7 stages that transform raw Supreme Court PDFs into a fine-tuned, queryable legal knowledge base:

```
COLLECT → EXTRACT → CHUNK → VECTOR-BUILD → CREATE-TRAINING-DATA → TRAIN-BGE → EVALUATE
```

### Stage 1: **Collect** - Download PDFs
- **Agent**: `DataCollectionAgent`
- **What it does**: Scrapes Sri Lankan Supreme Court website and downloads judgement PDFs
- **Source**: https://supremecourt.lk/wp-content/uploads/judgements/
- **Output**: 
  - `data/raw_pdfs/` - Downloaded PDF files
  - `data/pdf_metadata.csv` - Metadata including file hash, size, source URL

**Code**: `agents/data_collection_agent.py`

### Stage 2: **Extract** - Convert PDFs to Text
- **Agent**: `DocumentProcessingAgent`
- **What it does**: Extracts raw text from PDFs using OCR/text extraction
- **Output**: `data/processed_texts/` - Plain text documents from PDFs

**Code**: `agents/document_processing_agent.py`

### Stage 3: **Chunk** - Segment Text into Legal Chunks
- **Agent**: `ChunkingAgent`
- **What it does**: Splits long documents into meaningful legal segments (300+ character chunks)
- **Output**: `data/chunks/legal_chunks.json` - JSON with structure:
  ```json
  {
    "id": "chunk_123",
    "file_name": "judgment_001.pdf",
    "chunk_index": 0,
    "text": "Legal text content..."
  }
  ```

**Code**: `agents/chunking_agent.py`

### Stage 4: **Vector Build** - Create Vector Database
- **Agent**: `VectorAgent` / `VectorStorageAgent`
- **What it does**: 
  - Encodes text chunks using `BAAI/bge-m3` embedding model
  - Stores embeddings in ChromaDB for semantic search
- **Output**: `data/chroma_db/` - ChromaDB persistent storage with vector embeddings
- **Features**: Supports similarity search using cosine distance

**Code**: `agents/vector_agent.py`, `agents/vector_storage_agent.py`

### Stage 5: **Create Training Data** - Generate Query-Document Pairs
- **Agent**: `TrainingDataAgent`
- **What it does**:
  - Automatically categorizes legal chunks (Constitutional Law, Property Disputes, Employment, etc.)
  - Creates training pairs with query templates and positive/negative examples
  - Generates 500+ training samples for model fine-tuning
- **Output**: `data/training/retriever_train.jsonl` - JSONL training data:
  ```json
  {
    "query": "fundamental rights violation by police",
    "positive": "Legal chunk text...",
    "negative": "Different category chunk...",
    "category": "constitutional_law"
  }
  ```

**Code**: `agents/training_data_agent.py`

### Stage 6: **Train BGE** - Fine-tune Embedding Model
- **Agent**: `BGETrainingAgent`
- **What it does**:
  - Fine-tunes `BAAI/bge-m3` model on Sri Lankan legal data
  - Uses Multiple Negatives Ranking Loss for contrastive learning
  - Improves retrieval performance for legal documents
- **Output**: `models/bge-m3-srilanka-legal/` - Fine-tuned model checkpoint
- **Training**: 1 epoch, batch size 2, learning rate 2e-5

**Code**: `agents/bge_training_agent.py`, `train_bge_m3.py`

### Stage 7: **Evaluate** - Assess Retrieval Performance
- **Agent**: `EvaluationAgent` + `ComparisonAgent`
- **What it does**:
  - Evaluates base model vs. fine-tuned model on 8 test queries
  - Metrics: Hit@5 (does correct doc appear in top-5?), keyword support score
  - Compares performance improvements
- **Output**: 
  - `data/evaluation/base_bge_m3_result.json` - Base model evaluation
  - `data/evaluation/fine_tuned_bge_m3_result.json` - Fine-tuned model evaluation
  - `data/evaluation/comparison_result.json` - Side-by-side comparison

**Code**: `agents/evaluation_agent.py`, `agents/comparison_agent.py`

## 🚀 Quick Start

### Prerequisites
```bash
pip install -r requirements.txt
```

### Run Full Pipeline

**Option 1: Using FastAPI (Recommended for monitoring)**
```bash
python -m uvicorn app:app --reload
```
Then access `http://localhost:8000/docs` for interactive API

**Option 2: Direct execution**
```bash
python main_pipeline.py
```

### Run Individual Stages via API

```bash
# 1. Collect PDFs (with limit)
curl -X POST http://localhost:8000/pipeline/collect \
  -H "Content-Type: application/json" \
  -d '{"limit": 50}'

# 2. Extract text
curl -X POST http://localhost:8000/pipeline/extract

# 3. Chunk documents
curl -X POST http://localhost:8000/pipeline/chunk

# 4. Build vector database
curl -X POST http://localhost:8000/pipeline/vector-build

# 5. Create training data
curl -X POST http://localhost:8000/pipeline/create-training-data

# 6. Train fine-tuned model
curl -X POST http://localhost:8000/pipeline/train-bge

# 7. Evaluate models
curl -X POST http://localhost:8000/pipeline/evaluate

# 8. Compare base vs fine-tuned
curl -X GET http://localhost:8000/pipeline/compare

# Check job status
curl http://localhost:8000/status
```

### Query the Legal AI

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What are fundamental rights violations by police?"}'
```

## 📊 Data Structure

```
legal_ai_data_pipeline/
├── data/
│   ├── raw_pdfs/                    # Downloaded PDF files
│   ├── processed_texts/             # Extracted text files
│   ├── chunks/legal_chunks.json     # Segmented legal text
│   ├── training/
│   │   └── retriever_train.jsonl    # Query-document pairs
│   ├── evaluation/
│   │   ├── base_bge_m3_result.json
│   │   ├── fine_tuned_bge_m3_result.json
│   │   └── comparison_result.json
│   ├── chroma_db/                   # Vector database
│   └── pdf_metadata.csv             # Metadata tracking
├── models/
│   └── bge-m3-srilanka-legal/       # Fine-tuned embedding model
├── agents/                          # Agent implementations
├── app.py                           # FastAPI application
└── main_pipeline.py                 # Pipeline orchestration
```

## 🤖 Key Agents Explained

### DataCollectionAgent
Downloads PDFs from Supreme Court website with rate limiting and metadata tracking.

### DocumentProcessingAgent
Converts PDF files to plain text using industry-standard extraction libraries.

### ChunkingAgent
Intelligently segments documents with overlap to preserve context and maintain semantic coherence.

### VectorAgent
Manages vector storage and similarity search using ChromaDB and sentence transformers.

### TrainingDataAgent
Auto-categorizes documents and generates balanced training pairs for model fine-tuning:
- **Categories**: Constitutional Law, Fundamental Rights, Property Disputes, Labour Law, Commercial, Administrative
- **Strategy**: Generates query-positive-negative triplets for contrastive learning

### BGETrainingAgent
Fine-tunes the BAAI/bge-m3 embedding model (1024 dimensions) using:
- **Loss**: Multiple Negatives Ranking Loss
- **Optimization**: AdamW with gradient accumulation
- **Features**: CPU-safe training, no FP16 complications

### EvaluationAgent
Benchmarks retrieval quality on test queries:
- Hit@K: Does relevant document appear in top-K results?
- Keyword Support: How many query keywords appear in retrieved docs?

### ComparisonAgent
Produces side-by-side comparison of base vs. fine-tuned model performance.

## 📈 Legal Document Categories

The pipeline auto-categorizes documents into:
- **Constitutional Law**: Articles 12, 13, 14 violations, constitutional interpretation
- **Fundamental Rights**: Police violence, unlawful detention, torture
- **Property Disputes**: Land ownership, partition cases, property rights
- **Labour/Employment**: Termination, wages, employment disputes
- **Commercial**: Company disputes, commercial appeals, contracts
- **Administrative**: Legitimate expectation, administrative decisions, natural justice

## 🔄 Pipeline Flow Example

```
Input: Supreme Court judgement PDF
    ↓
COLLECT: Download "SC_FR_123_2015.pdf"
    ↓
EXTRACT: Convert PDF → Text (2500 characters)
    ↓
CHUNK: Split into 8-10 chunks (300+ chars each)
    ↓
VECTOR-BUILD: Encode chunks → store in ChromaDB
    ↓
CREATE-TRAINING: Label chunks → generate query pairs
    ↓
TRAIN-BGE: Fine-tune embedding model on 500 pairs
    ↓
EVALUATE: Test "fundamental rights violation" query
    ↓
OUTPUT: Hit@5=6/8 (75%), improvement confirmed!
```

## 🔧 Configuration

### Training Parameters (agents/bge_training_agent.py)
- **Base Model**: BAAI/bge-m3
- **Epochs**: 1
- **Batch Size**: 2 (CPU-optimized)
- **Learning Rate**: 2e-5
- **Gradient Accumulation**: 8 steps
- **Warmup Ratio**: 10%

### Chunking Parameters (agents/chunking_agent.py)
- **Min Chunk Size**: 300 characters
- **Overlap**: Configurable context preservation

### Evaluation Parameters (agents/evaluation_agent.py)
- **Top-K**: 5 (retrieve top 5 documents)
- **Min Keywords**: 2 out of N query keywords must match

## 📝 API Endpoints

| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/` | Home info & pipeline stages |
| GET | `/status` | Current job status |
| POST | `/ask` | Ask legal question |
| GET | `/pipeline/compare` | Compare model performance |
| POST | `/pipeline/collect` | Download PDFs |
| POST | `/pipeline/extract` | Extract text from PDFs |
| POST | `/pipeline/chunk` | Create text chunks |
| POST | `/pipeline/vector-build` | Build vector database |
| POST | `/pipeline/create-training-data` | Generate training pairs |
| POST | `/pipeline/train-bge` | Train fine-tuned model |
| POST | `/pipeline/evaluate` | Evaluate retrieval |

## ⚠️ Important Notes

1. **First Run**: The first `/pipeline/collect` may take 30+ minutes depending on PDF count
2. **Storage**: Full pipeline with ~1000 documents requires ~2-3 GB disk space
3. **GPU/CPU**: Fine-tuning runs on CPU by default (safe for most environments)
4. **Rate Limiting**: Collection agent includes 1-second delay between downloads
5. **Legal Disclaimer**: This system retrieves judgements; it's not formal legal advice

## 🎯 Use Cases

- **Legal Research**: Semantic search across Supreme Court judgements
- **Case Law Discovery**: Find similar legal precedents
- **Legal Document Classification**: Categorize judgements by topic
- **AI Training**: Fine-tune models on domain-specific legal data
- **Academic Research**: Analyze patterns in Sri Lankan legal decisions

## 📚 Model Information

**Base Model**: BAAI/bge-m3
- 1024-dimensional embeddings
- Multilingual (supports 111+ languages)
- Optimized for retrieval-augmented generation

**Fine-Tuned Model**: bge-m3-srilanka-legal
- Specialized for Sri Lankan legal terminology
- 500+ training samples from Supreme Court judgements
- ~5-10% performance improvement on legal queries

## 🤝 Contributing

To extend the pipeline:
1. Create new agents in `agents/` directory
2. Follow the existing agent pattern (initialize, run, return status)
3. Integrate with FastAPI in `app.py`
4. Add to pipeline stages in `main_pipeline.py`

## 📄 License

Research use. See LICENSE for details.

---

**Built with**: Python, FastAPI, Sentence Transformers, ChromaDB, BAAI/BGE-M3
