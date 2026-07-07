# agents/comparison_agent.py

import os
import json


class ComparisonAgent:
    def __init__(
        self,
        base_path="data/evaluation/base_bge_m3_result.json",
        fine_tuned_path="data/evaluation/fine_tuned_bge_m3_result.json",
        output_path="data/evaluation/comparison_result.json"
    ):
        self.base_path = base_path
        self.fine_tuned_path = fine_tuned_path
        self.output_path = output_path

    def load_json(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing file: {path}")

        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def compare(self):
        base = self.load_json(self.base_path)
        fine = self.load_json(self.fine_tuned_path)

        base_hit = base.get("hit_at_k", 0)
        fine_hit = fine.get("hit_at_k", 0)

        base_keyword = base.get("average_keyword_support", 0)
        fine_keyword = fine.get("average_keyword_support", 0)

        result = {
            "base_model": {
                "file": self.base_path,
                "hit_at_k": base_hit,
                "average_keyword_support": base_keyword
            },
            "fine_tuned_model": {
                "file": self.fine_tuned_path,
                "hit_at_k": fine_hit,
                "average_keyword_support": fine_keyword
            },
            "improvement": {
                "hit_at_k_difference": fine_hit - base_hit,
                "average_keyword_support_difference": fine_keyword - base_keyword
            },
            "conclusion": self.make_conclusion(
                base_hit,
                fine_hit,
                base_keyword,
                fine_keyword
            )
        }

        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)

        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=4, ensure_ascii=False)

        return result

    def make_conclusion(self, base_hit, fine_hit, base_keyword, fine_keyword):
        if fine_hit > base_hit and fine_keyword >= base_keyword:
            return "The fine-tuned BGE-M3 retriever improved retrieval performance over the base BGE-M3 model."
        elif fine_hit == base_hit and fine_keyword > base_keyword:
            return "The fine-tuned BGE-M3 retriever preserved Hit@K and improved keyword support."
        elif fine_hit < base_hit:
            return "The fine-tuned model reduced retrieval performance. The training data may need cleaning or better negative sampling."
        else:
            return "The fine-tuned model shows similar performance to the base model."