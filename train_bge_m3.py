# train_bge_m3.py

import json
import os

# Force CPU before torch/model loading
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from datasets import Dataset
from sentence_transformers import SentenceTransformer
from sentence_transformers.sentence_transformer.losses import MultipleNegativesRankingLoss
from sentence_transformers import (
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)


TRAIN_PATH = "data/training/retriever_train.jsonl"
MODEL_NAME = "BAAI/bge-m3"
OUTPUT_DIR = "models/bge-m3-srilanka-legal"


def load_training_data(path):
    rows = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)

            rows.append({
                "anchor": item["query"],
                "positive": item["positive"]
            })

    return Dataset.from_list(rows)


def main():
    train_dataset = load_training_data(TRAIN_PATH)

    # Load model directly on CPU
    model = SentenceTransformer(MODEL_NAME, device="cpu")

    loss = MultipleNegativesRankingLoss(model)

    args = SentenceTransformerTrainingArguments(
        output_dir=OUTPUT_DIR,

        # CPU-safe settings
        fp16=False,
        bf16=False,
        dataloader_pin_memory=False,

        # Training settings
        num_train_epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=2e-5,

        # 10% of 5015 total steps
        warmup_steps=502,

        logging_steps=10,
        save_strategy="epoch",
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        loss=loss,
    )

    trainer.train()

    model.save(OUTPUT_DIR)

    print(f"Fine-tuned CPU model saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()