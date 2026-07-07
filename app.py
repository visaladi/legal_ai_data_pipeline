import os
import json
from datetime import datetime
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel

from agents.data_collection_agent import DataCollectionAgent
from agents.document_processing_agent import DocumentProcessingAgent
from agents.chunking_agent import ChunkingAgent
from agents.vector_agent import VectorAgent
from agents.legal_response_agent import LegalResponseAgent
from agents.training_data_agent import TrainingDataAgent
from agents.bge_training_agent import BGETrainingAgent
from agents.evaluation_agent import EvaluationAgent
from agents.comparison_agent import ComparisonAgent

SUPREME_COURT_INDEX = "https://supremecourt.lk/wp-content/uploads/judgements/"
STATUS_PATH = "data/job_status.json"

app = FastAPI(title="Sri Lankan Legal AI Assistant Research API")

legal_agent = LegalResponseAgent()


class LegalQuery(BaseModel):
    question: str


class CollectRequest(BaseModel):
    limit: int | None = None


class TrainingDataRequest(BaseModel):
    max_pairs: int | None = None
    min_text_length: int = 300


class TrainRequest(BaseModel):
    epochs: int = 1
    batch_size: int = 2
    learning_rate: float = 2e-5
    fp16: bool = False


class EvaluateRequest(BaseModel):
    top_k: int = 5


def ensure_data_dir():
    os.makedirs("data", exist_ok=True)


def update_status(job_name: str, status: str, message: str = "", result=None):
    ensure_data_dir()

    payload = {
        "job_name": job_name,
        "status": status,
        "message": message,
        "result": result,
        "updated_at": datetime.now().isoformat()
    }

    with open(STATUS_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4, ensure_ascii=False)


@app.get("/")
def home():
    return {
        "message": "Sri Lankan Legal AI Assistant Research API is running",
        "pipeline": [
            "collect",
            "extract",
            "chunk",
            "vector-build",
            "create-training-data",
            "train-bge",
            "evaluate"
        ],
        "docs": "Open /docs"
    }


@app.get("/status")
def get_status():
    if not os.path.exists(STATUS_PATH):
        return {"status": "no_job_started"}

    with open(STATUS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@app.post("/ask")
def ask_legal_question(query: LegalQuery):
    answer = legal_agent.answer_without_llm(query.question)

    return {
        "question": query.question,
        "answer": answer
    }


def collect_job(limit=None):
    try:
        update_status("collect", "running", "Downloading Supreme Court PDFs")
        collector = DataCollectionAgent()
        downloaded = collector.run(SUPREME_COURT_INDEX, limit=limit)

        update_status(
            "collect",
            "completed",
            f"Downloaded/found {len(downloaded)} PDFs",
            {"downloaded_count": len(downloaded)}
        )

    except Exception as e:
        update_status("collect", "failed", str(e))


@app.post("/pipeline/collect")
def collect_pdfs(request: CollectRequest, background_tasks: BackgroundTasks):
    background_tasks.add_task(collect_job, request.limit)

    return {
        "status": "started",
        "job": "collect",
        "message": "Check /status"
    }


def extract_job():
    try:
        update_status("extract", "running", "Extracting text from PDFs")
        processor = DocumentProcessingAgent()
        processor.run()

        update_status("extract", "completed", "Text extraction completed")

    except Exception as e:
        update_status("extract", "failed", str(e))

@app.get("/pipeline/compare")
def compare_base_and_finetuned():
    agent = ComparisonAgent()
    result = agent.compare()

    return result

@app.post("/pipeline/extract")
def extract_text(background_tasks: BackgroundTasks):
    background_tasks.add_task(extract_job)

    return {
        "status": "started",
        "job": "extract",
        "message": "Check /status"
    }


def chunk_job():
    try:
        update_status("chunk", "running", "Creating legal chunks")
        chunker = ChunkingAgent()
        chunker.run()

        update_status("chunk", "completed", "Chunking completed")

    except Exception as e:
        update_status("chunk", "failed", str(e))


@app.post("/pipeline/chunk")
def chunk_text(background_tasks: BackgroundTasks):
    background_tasks.add_task(chunk_job)

    return {
        "status": "started",
        "job": "chunk",
        "message": "Check /status"
    }


def vector_build_job():
    try:
        update_status("vector_build", "running", "Building BGE-M3 vector index")
        vector_agent = VectorAgent()
        vector_agent.build_index()

        update_status("vector_build", "completed", "Vector index created")

    except Exception as e:
        update_status("vector_build", "failed", str(e))


@app.post("/pipeline/vector-build")
def build_vector_db(background_tasks: BackgroundTasks):
    background_tasks.add_task(vector_build_job)

    return {
        "status": "started",
        "job": "vector_build",
        "message": "Check /status"
    }


def training_data_job(max_pairs=None, min_text_length=300):
    try:
        update_status("training_data", "running", "Creating retriever training data")
        agent = TrainingDataAgent()

        result = agent.create_training_pairs(
            max_pairs=max_pairs,
            min_text_length=min_text_length
        )

        update_status(
            "training_data",
            "completed",
            f"Created {result['total_pairs']} training pairs",
            result
        )

    except Exception as e:
        update_status("training_data", "failed", str(e))


@app.post("/pipeline/create-training-data")
def create_training_data(
    request: TrainingDataRequest,
    background_tasks: BackgroundTasks
):
    background_tasks.add_task(
        training_data_job,
        request.max_pairs,
        request.min_text_length
    )

    return {
        "status": "started",
        "job": "training_data",
        "message": "Check /status"
    }


@app.get("/training/preview")
def preview_training_data(limit: int = 5):
    agent = TrainingDataAgent()
    rows = agent.preview(limit=limit)

    return {
        "count": len(rows),
        "rows": rows
    }


def train_bge_job(epochs=1, batch_size=2, learning_rate=2e-5, fp16=False):
    try:
        update_status("train_bge", "running", "Fine-tuning BGE-M3")
        trainer = BGETrainingAgent()

        result = trainer.train(
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            fp16=fp16
        )

        update_status(
            "train_bge",
            "completed",
            "BGE-M3 fine-tuning completed",
            result
        )

    except Exception as e:
        update_status("train_bge", "failed", str(e))


@app.post("/pipeline/train-bge")
def train_bge(request: TrainRequest, background_tasks: BackgroundTasks):
    background_tasks.add_task(
        train_bge_job,
        request.epochs,
        request.batch_size,
        request.learning_rate,
        request.fp16
    )

    return {
        "status": "started",
        "job": "train_bge",
        "message": "Check /status"
    }


def evaluate_job(top_k=5):
    try:
        update_status("evaluate", "running", "Evaluating retrieval performance")
        evaluator = EvaluationAgent()
        result = evaluator.evaluate(top_k=top_k)

        update_status(
            "evaluate",
            "completed",
            "Evaluation completed",
            {
                "hit_at_k": result["hit_at_k"],
                "average_keyword_support": result["average_keyword_support"],
                "output_path": "data/evaluation/evaluation_result.json"
            }
        )

    except Exception as e:
        update_status("evaluate", "failed", str(e))


@app.post("/pipeline/evaluate")
def evaluate(request: EvaluateRequest, background_tasks: BackgroundTasks):
    background_tasks.add_task(evaluate_job, request.top_k)

    return {
        "status": "started",
        "job": "evaluate",
        "message": "Check /status"
    }


def run_all_job(limit=None, max_pairs=None):
    try:
        update_status("run_all", "running", "Step 1/7: Collecting PDFs")
        collector = DataCollectionAgent()
        collector.run(SUPREME_COURT_INDEX, limit=limit)

        update_status("run_all", "running", "Step 2/7: Extracting text")
        processor = DocumentProcessingAgent()
        processor.run()

        update_status("run_all", "running", "Step 3/7: Chunking text")
        chunker = ChunkingAgent()
        chunker.run()

        update_status("run_all", "running", "Step 4/7: Building vector DB")
        vector_agent = VectorAgent()
        vector_agent.build_index()

        update_status("run_all", "running", "Step 5/7: Creating training data")
        training_agent = TrainingDataAgent()
        train_result = training_agent.create_training_pairs(max_pairs=max_pairs)

        update_status("run_all", "running", "Step 6/7: Evaluating base retriever")
        evaluator = EvaluationAgent()
        eval_result = evaluator.evaluate(top_k=5)

        update_status(
            "run_all",
            "completed",
            "Pipeline completed. Training data created and base retriever evaluated.",
            {
                "training_pairs": train_result["total_pairs"],
                "hit_at_5": eval_result["hit_at_k"],
                "average_keyword_support": eval_result["average_keyword_support"]
            }
        )

    except Exception as e:
        update_status("run_all", "failed", str(e))


@app.post("/pipeline/run-all")
def run_all(request: CollectRequest, background_tasks: BackgroundTasks):
    background_tasks.add_task(run_all_job, request.limit, None)

    return {
        "status": "started",
        "job": "run_all",
        "message": "Check /status"
    }