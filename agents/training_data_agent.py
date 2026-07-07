import os
import json
import random
from typing import Dict, List, Optional


class TrainingDataAgent:
    def __init__(
        self,
        chunks_path: str = "data/chunks/legal_chunks.json",
        output_path: str = "data/training/retriever_train.jsonl"
    ):
        self.chunks_path = chunks_path
        self.output_path = output_path
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)

        self.category_keywords = {
            "fundamental_rights": [
                "fundamental rights", "article 11", "article 12",
                "article 13", "article 14", "constitution", "sc fr"
            ],
            "police_arrest": [
                "police", "arrest", "unlawful arrest",
                "inspector general of police", "detention", "custody"
            ],
            "detention": [
                "detention", "detained", "custody", "remand", "article 13"
            ],
            "torture": [
                "torture", "cruel", "inhuman", "degrading treatment",
                "assault", "article 11"
            ],
            "property_dispute": [
                "land", "property", "partition", "deed",
                "title", "possession", "ownership"
            ],
            "labour_dispute": [
                "employment", "employee", "employer", "labour",
                "termination", "service", "dismissal"
            ],
            "contract_dispute": [
                "contract", "agreement", "breach", "damages", "obligation"
            ],
            "commercial_appeal": [
                "commercial", "company", "plaintiff", "defendant",
                "commercial high court", "appeal"
            ],
            "administrative_law": [
                "public service commission", "administrative",
                "natural justice", "legitimate expectation",
                "public authority", "writ"
            ],
            "constitutional_law": [
                "constitution", "article", "jurisdiction",
                "fundamental rights", "constitutional"
            ]
        }

        self.query_templates = {
            "fundamental_rights": [
                "Supreme Court cases about fundamental rights violations",
                "Article 12 equality violation Supreme Court Sri Lanka",
                "fundamental rights application under the Sri Lankan Constitution"
            ],
            "police_arrest": [
                "Supreme Court cases about unlawful arrest by police",
                "police misconduct and fundamental rights in Sri Lanka",
                "police arrest and detention fundamental rights case"
            ],
            "detention": [
                "Supreme Court cases about unlawful detention",
                "Article 13 detention and arrest Sri Lanka",
                "fundamental rights violation due to detention"
            ],
            "torture": [
                "Article 11 torture Supreme Court Sri Lanka",
                "fundamental rights cases about torture by police",
                "cruel inhuman degrading treatment Sri Lanka case"
            ],
            "property_dispute": [
                "Supreme Court cases about land ownership dispute",
                "property dispute appeal Sri Lanka",
                "partition case Supreme Court Sri Lanka"
            ],
            "labour_dispute": [
                "Supreme Court employment termination case Sri Lanka",
                "labour dispute and dismissal case",
                "employee service dispute Supreme Court"
            ],
            "contract_dispute": [
                "Supreme Court breach of contract case",
                "contract dispute appeal Sri Lanka",
                "agreement and damages Supreme Court case"
            ],
            "commercial_appeal": [
                "commercial appeal Supreme Court Sri Lanka",
                "company dispute Supreme Court case",
                "commercial high court appeal"
            ],
            "administrative_law": [
                "administrative law legitimate expectation Supreme Court",
                "public authority arbitrary decision case",
                "natural justice administrative decision Sri Lanka"
            ],
            "constitutional_law": [
                "constitutional law Supreme Court Sri Lanka",
                "interpretation of the Constitution by Supreme Court",
                "constitutional jurisdiction Sri Lanka"
            ]
        }

    def load_chunks(self) -> List[Dict]:
        if not os.path.exists(self.chunks_path):
            raise FileNotFoundError(
                f"Missing {self.chunks_path}. Run collect, extract, and chunk first."
            )

        with open(self.chunks_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def detect_categories(self, text: str) -> List[str]:
        lower = text.lower()
        categories = []

        for category, keywords in self.category_keywords.items():
            score = sum(1 for kw in keywords if kw in lower)

            if score >= 2:
                categories.append(category)

        return categories

    def label_chunks(self, min_text_length: int = 300) -> List[Dict]:
        chunks = self.load_chunks()
        labeled = []

        for chunk in chunks:
            text = chunk.get("text", "")

            if len(text) < min_text_length:
                continue

            categories = self.detect_categories(text)

            for category in categories:
                labeled.append({
                    "id": chunk["id"],
                    "file_name": chunk["file_name"],
                    "chunk_index": chunk["chunk_index"],
                    "text": text,
                    "category": category
                })

        return labeled

    def create_training_pairs(
        self,
        max_pairs: Optional[int] = None,
        min_text_length: int = 300
    ) -> Dict:
        labeled = self.label_chunks(min_text_length=min_text_length)

        if not labeled:
            raise ValueError("No labeled chunks found. Check extracted text/chunks.")

        rows = []

        for positive in labeled:
            category = positive["category"]

            negatives = [
                item for item in labeled
                if item["category"] != category
            ]

            if not negatives:
                continue

            negative = random.choice(negatives)
            query = random.choice(self.query_templates[category])

            rows.append({
                "query": query,
                "positive": positive["text"],
                "negative": negative["text"],
                "category": category,
                "positive_file": positive["file_name"],
                "positive_chunk": positive["chunk_index"],
                "negative_file": negative["file_name"],
                "negative_chunk": negative["chunk_index"]
            })

        random.shuffle(rows)

        if max_pairs is not None:
            rows = rows[:max_pairs]

        with open(self.output_path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        category_counts = {}

        for row in rows:
            category_counts[row["category"]] = category_counts.get(row["category"], 0) + 1

        return {
            "output_path": self.output_path,
            "total_pairs": len(rows),
            "category_counts": category_counts
        }

    def preview(self, limit: int = 5) -> List[Dict]:
        if not os.path.exists(self.output_path):
            raise FileNotFoundError("Training file not found. Create training data first.")

        rows = []

        with open(self.output_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= limit:
                    break

                item = json.loads(line)

                rows.append({
                    "query": item["query"],
                    "category": item["category"],
                    "positive_file": item["positive_file"],
                    "positive_chunk": item["positive_chunk"],
                    "negative_file": item["negative_file"],
                    "negative_chunk": item["negative_chunk"],
                    "positive_preview": item["positive"][:400],
                    "negative_preview": item["negative"][:400]
                })

        return rows