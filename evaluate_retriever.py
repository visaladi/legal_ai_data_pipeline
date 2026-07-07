# evaluate_retriever.py

import json
from agents.vector_agent import VectorAgent


EVAL_PATH = "data/eval/retrieval_eval.json"


def contains_keywords(text, keywords):
    lower = text.lower()
    matched = 0

    for kw in keywords:
        if kw.lower() in lower:
            matched += 1

    return matched


def main():
    vector_agent = VectorAgent()

    with open(EVAL_PATH, "r", encoding="utf-8") as f:
        eval_items = json.load(f)

    total = 0
    hit_at_5 = 0
    keyword_scores = []

    for item in eval_items:
        query = item["query"]
        keywords = item["relevant_keywords"]

        results = vector_agent.search(query, top_k=5)

        docs = results["documents"][0]

        found = False
        best_score = 0

        for doc in docs:
            score = contains_keywords(doc, keywords)
            best_score = max(best_score, score)

            if score >= 2:
                found = True

        total += 1

        if found:
            hit_at_5 += 1

        keyword_scores.append(best_score / len(keywords))

        print("\nQuery:", query)
        print("Found relevant in top 5:", found)
        print("Best keyword score:", best_score)

    print("\nEvaluation Results")
    print("Hit@5:", hit_at_5 / total)
    print("Average Keyword Match:", sum(keyword_scores) / len(keyword_scores))


if __name__ == "__main__":
    main()