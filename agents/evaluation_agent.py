import os
import json
from typing import List, Dict
from agents.vector_agent import VectorAgent


class EvaluationAgent:
    def __init__(
        self,
        eval_path: str = "data/evaluation/retrieval_eval.json",
        output_path: str = "data/evaluation/evaluation_result.json"
    ):
        self.eval_path = eval_path
        self.output_path = output_path
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)

    def create_default_eval_set(self):
        os.makedirs(os.path.dirname(self.eval_path), exist_ok=True)

        eval_data = [
            {
                "query": "fundamental rights violation by police",
                "relevant_keywords": ["fundamental rights", "police"]
            },
            {
                "query": "unlawful arrest by police officers",
                "relevant_keywords": ["arrest", "police"]
            },
            {
                "query": "Article 13 unlawful detention case",
                "relevant_keywords": ["article 13", "detention"]
            },
            {
                "query": "Article 11 torture by police",
                "relevant_keywords": ["article 11", "torture"]
            },
            {
                "query": "land ownership property dispute",
                "relevant_keywords": ["land", "property"]
            },
            {
                "query": "employment termination labour dispute",
                "relevant_keywords": ["employment", "termination"]
            },
            {
                "query": "commercial appeal company dispute",
                "relevant_keywords": ["commercial", "company"]
            },
            {
                "query": "administrative law legitimate expectation",
                "relevant_keywords": ["legitimate expectation", "administrative"]
            }
        ]

        with open(self.eval_path, "w", encoding="utf-8") as f:
            json.dump(eval_data, f, indent=4, ensure_ascii=False)

        return {
            "status": "created",
            "eval_path": self.eval_path,
            "total_queries": len(eval_data)
        }

    def load_eval_set(self) -> List[Dict]:
        if not os.path.exists(self.eval_path):
            self.create_default_eval_set()

        with open(self.eval_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def keyword_score(self, text: str, keywords: List[str]) -> int:
        lower = text.lower()
        return sum(1 for kw in keywords if kw.lower() in lower)

    def evaluate(self, top_k: int = 5):
        vector_agent = VectorAgent()
        eval_items = self.load_eval_set()

        total = len(eval_items)
        hit_at_k = 0
        average_keyword_support = 0.0
        detailed_results = []

        for item in eval_items:
            query = item["query"]
            keywords = item["relevant_keywords"]

            results = vector_agent.search(query, top_k=top_k)
            documents = results["documents"][0]
            metadatas = results["metadatas"][0]

            found = False
            best_score = 0

            returned = []

            for doc, meta in zip(documents, metadatas):
                score = self.keyword_score(doc, keywords)
                best_score = max(best_score, score)

                if score >= min(2, len(keywords)):
                    found = True

                returned.append({
                    "file_name": meta["file_name"],
                    "chunk_index": meta["chunk_index"],
                    "keyword_score": score,
                    "preview": doc[:300]
                })

            if found:
                hit_at_k += 1

            average_keyword_support += best_score / max(len(keywords), 1)

            detailed_results.append({
                "query": query,
                "keywords": keywords,
                "found_relevant": found,
                "best_keyword_score": best_score,
                "returned": returned
            })

        result = {
            "top_k": top_k,
            "total_queries": total,
            "hit_at_k": hit_at_k / total if total else 0,
            "average_keyword_support": average_keyword_support / total if total else 0,
            "details": detailed_results
        }

        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=4, ensure_ascii=False)

        return result