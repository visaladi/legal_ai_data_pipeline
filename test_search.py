# test_search.py

from agents.vector_storage_agent import VectorStorageAgent

vector_agent = VectorStorageAgent()

query = "fundamental rights violation by police in Sri Lanka"

results = vector_agent.search(query, top_k=5)

for i, doc in enumerate(results["documents"][0]):
    print("\n====================")
    print(f"Result {i + 1}")
    print("====================")
    print(doc[:1000])