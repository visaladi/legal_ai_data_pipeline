# main_pipeline.py

from agents.data_collection_agent import DataCollectionAgent
from agents.document_processing_agent import DocumentProcessingAgent
from agents.chunking_agent import ChunkingAgent
from agents.vector_agent import VectorAgent


SUPREME_COURT_INDEX = "https://supremecourt.lk/wp-content/uploads/judgements/"


def main():
    print("Step 1: Downloading Supreme Court PDFs")
    collector = DataCollectionAgent()
    collector.run(SUPREME_COURT_INDEX, limit=None)

    print("Step 2: Extracting PDF text")
    processor = DocumentProcessingAgent()
    processor.run()

    print("Step 3: Chunking legal text")
    chunker = ChunkingAgent()
    chunker.run()

    print("Step 4: Creating vector database")
    vector_agent = VectorAgent()
    vector_agent.build_index()

    print("Pipeline completed.")


if __name__ == "__main__":
    main()