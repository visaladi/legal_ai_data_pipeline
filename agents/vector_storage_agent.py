# agents/vector_storage_agent.py

import json
import chromadb
from tqdm import tqdm
from sentence_transformers import SentenceTransformer


class VectorStorageAgent:
    def __init__(
        self,
        chunks_path="data/chunks/legal_chunks.json",
        db_path="data/chroma_db",
        collection_name="sri_lankan_legal_docs"
    ):
        self.chunks_path = chunks_path
        self.db_path = db_path
        self.collection_name = collection_name

        self.model = SentenceTransformer("BAAI/bge-m3")

        self.client = chromadb.PersistentClient(path=self.db_path)

        self.collection = self.client.get_or_create_collection(
            name=self.collection_name
        )

    def load_chunks(self):
        with open(self.chunks_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def store_chunks(self):
        chunks = self.load_chunks()

        for item in tqdm(chunks, desc="Storing vectors"):
            embedding = self.model.encode(item["text"]).tolist()

            self.collection.add(
                ids=[item["chunk_id"]],
                embeddings=[embedding],
                documents=[item["text"]],
                metadatas=[{
                    "file_name": item["file_name"],
                    "chunk_index": item["chunk_index"]
                }]
            )

        print("Vector storage completed.")

    def search(self, query: str, top_k=5):
        query_embedding = self.model.encode(query).tolist()

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )

        return results