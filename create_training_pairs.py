# create_training_pairs.py

import json
import random


CHUNKS_PATH = "data/chunks/legal_chunks.json"
OUTPUT_PATH = "data/training/retriever_train.jsonl"


QUERY_TEMPLATES = {
    "fundamental_rights": [
        "fundamental rights violation",
        "Article 12 equality violation",
        "Article 13 unlawful arrest",
        "police fundamental rights case",
        "violation of constitutional rights by state officers"
    ],
    "police": [
        "police arrest detention case",
        "police misconduct Supreme Court",
        "unlawful arrest by police",
        "police custody fundamental rights"
    ],
    "commercial": [
        "commercial dispute appeal",
        "company contract dispute",
        "civil commercial appeal Supreme Court"
    ],
    "property": [
        "land dispute appeal",
        "property ownership dispute",
        "partition land case"
    ]
}


def detect_label(text):
    lower = text.lower()

    if "fundamental rights" in lower or "article 12" in lower or "article 13" in lower:
        return "fundamental_rights"

    if "police" in lower or "inspector general of police" in lower:
        return "police"

    if "commercial" in lower or "contract" in lower or "company" in lower:
        return "commercial"

    if "land" in lower or "property" in lower or "partition" in lower:
        return "property"

    return None


def main():
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    labeled = []

    for item in chunks:
        label = detect_label(item["text"])

        if label:
            labeled.append({
                "label": label,
                "text": item["text"],
                "file_name": item["file_name"],
                "chunk_index": item["chunk_index"]
            })

    with open(OUTPUT_PATH, "w", encoding="utf-8") as out:
        for item in labeled:
            query = random.choice(QUERY_TEMPLATES[item["label"]])

            negatives = [
                x for x in labeled
                if x["label"] != item["label"]
            ]

            if not negatives:
                continue

            negative = random.choice(negatives)

            row = {
                "query": query,
                "positive": item["text"],
                "negative": negative["text"],
                "positive_file": item["file_name"],
                "positive_chunk": item["chunk_index"],
                "negative_file": negative["file_name"],
                "negative_chunk": negative["chunk_index"]
            }

            out.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Training pairs saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()