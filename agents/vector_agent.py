# agents/vector_agent.py

import json
import chromadb
from sentence_transformers import SentenceTransformer


class VectorAgent:
    def __init__(
        self,
        chunks_path="data/chunks/legal_chunks.json",
        db_path="data/chroma_db",
        collection_name="supreme_court_judgments",
        #model_name="BAAI/bge-m3"
        model_name = "models/bge-m3-srilanka-legal"
        #model_name = "models/bge-m3-srilanka-legal-fast"
    ):
        self.chunks_path = chunks_path
        self.model = SentenceTransformer(model_name)

        self.client = chromadb.PersistentClient(path=db_path)

        self.collection = self.client.get_or_create_collection(
            name=collection_name
        )

    def load_chunks(self):
        with open(self.chunks_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def build_index(self, batch_size=32):
        chunks = self.load_chunks()

        ids = [item["id"] for item in chunks]
        documents = [item["text"] for item in chunks]
        metadatas = [
            {
                "file_name": item["file_name"],
                "chunk_index": item["chunk_index"]
            }
            for item in chunks
        ]

        for start in range(0, len(chunks), batch_size):
            end = start + batch_size

            batch_ids = ids[start:end]
            batch_docs = documents[start:end]
            batch_meta = metadatas[start:end]

            embeddings = self.model.encode(
                batch_docs,
                batch_size=batch_size,
                normalize_embeddings=True
            ).tolist()

            self.collection.upsert(
                ids=batch_ids,
                documents=batch_docs,
                embeddings=embeddings,
                metadatas=batch_meta
            )

            print(f"Indexed {end}/{len(chunks)} chunks")

        print("Vector index created successfully.")

    def search(self, query: str, top_k=5):
        query_embedding = self.model.encode(
            query,
            normalize_embeddings=True
        ).tolist()

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )

        return results