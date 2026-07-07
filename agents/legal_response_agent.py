# agents/legal_response_agent.py

from agents.retrieval_agent import RetrievalAgent


class LegalResponseAgent:
    def __init__(self):
        self.retrieval_agent = RetrievalAgent()

    def create_context(self, retrieved_docs):
        context = ""

        for i, item in enumerate(retrieved_docs, start=1):
            context += f"\nSOURCE {i}\n"
            context += f"File: {item['file_name']}\n"
            context += f"Chunk: {item['chunk_index']}\n"
            context += item["text"]
            context += "\n"

        return context

    def answer_without_llm(self, query: str):
        retrieved_docs = self.retrieval_agent.retrieve(query)

        context = self.create_context(retrieved_docs)

        response = f"""
This is a retrieval-based legal information response.

User Question:
{query}

Relevant Supreme Court Judgment Extracts:
{context}

Important Disclaimer:
This system provides general legal information based on retrieved Sri Lankan Supreme Court documents. It is not formal legal advice. A qualified lawyer should review the matter.
"""
        return response