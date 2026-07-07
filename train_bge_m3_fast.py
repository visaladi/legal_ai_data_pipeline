# train_bge_m3_fast.py

import json
from datasets import Dataset
from sentence_transformers import SentenceTransformer
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers import (
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments
)


TRAIN_PATH = "data/training/retriever_train.jsonl"
MODEL_NAME = "BAAI/bge-m3"
OUTPUT_DIR = "models/bge-m3-srilanka-legal-fast"


def load_training_data(path, limit=500):
    rows = []

    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= limit:
                break

            item = json.loads(line)

            rows.append({
                "anchor": item["query"],
                "positive": item["positive"]
            })

    return Dataset.from_list(rows)


def main():
    train_dataset = load_training_data(TRAIN_PATH, limit=500)

    print(f"Training rows: {len(train_dataset)}")

    model = SentenceTransformer(MODEL_NAME)
    loss = MultipleNegativesRankingLoss(model)

    args = SentenceTransformerTrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        learning_rate=2e-5,
        warmup_ratio=0.05,
        fp16=False,
        logging_steps=10,
        save_strategy="steps",
        save_steps=100,
        max_steps=300
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        loss=loss
    )

    trainer.train()
    model.save(OUTPUT_DIR)

    print(f"Fast fine-tuned model saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()