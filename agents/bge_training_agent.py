import os
import json
from datasets import Dataset
from sentence_transformers import SentenceTransformer
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers import (
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments
)


class BGETrainingAgent:
    def __init__(
        self,
        train_path: str = "data/training/retriever_train.jsonl",
        base_model: str = "BAAI/bge-m3",
        output_dir: str = "models/bge-m3-srilanka-legal"
    ):
        self.train_path = train_path
        self.base_model = base_model
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def load_dataset(self):
        if not os.path.exists(self.train_path):
            raise FileNotFoundError(
                f"Training file not found: {self.train_path}. Create training data first."
            )

        rows = []

        with open(self.train_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)

                rows.append({
                    "anchor": item["query"],
                    "positive": item["positive"]
                })

        if not rows:
            raise ValueError("Training dataset is empty.")

        return Dataset.from_list(rows)

    def train(
        self,
        epochs: int = 1,
        batch_size: int = 2,
        learning_rate: float = 2e-5,
        fp16: bool = False
    ):
        dataset = self.load_dataset()

        model = SentenceTransformer(self.base_model)
        loss = MultipleNegativesRankingLoss(model)

        args = SentenceTransformerTrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=8,
            learning_rate=learning_rate,
            warmup_ratio=0.1,
            fp16=fp16,
            logging_steps=10,
            save_strategy="epoch"
        )

        trainer = SentenceTransformerTrainer(
            model=model,
            args=args,
            train_dataset=dataset,
            loss=loss
        )

        trainer.train()
        model.save(self.output_dir)

        return {
            "status": "completed",
            "base_model": self.base_model,
            "output_dir": self.output_dir,
            "epochs": epochs,
            "batch_size": batch_size,
            "training_rows": len(dataset)
        }