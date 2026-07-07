# agents/document_processing_agent.py

import os
import fitz


class DocumentProcessingAgent:
    def __init__(
        self,
        input_dir="data/raw_pdfs",
        output_dir="data/extracted_text"
    ):
        self.input_dir = input_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def extract_text(self, pdf_path: str):
        text = ""

        doc = fitz.open(pdf_path)

        for page_no, page in enumerate(doc, start=1):
            page_text = page.get_text()
            text += f"\n\n--- PAGE {page_no} ---\n"
            text += page_text

        return text

    def clean_text(self, text: str):
        lines = text.splitlines()
        cleaned = []

        for line in lines:
            line = line.strip()

            if not line:
                continue

            if line.lower().startswith("page "):
                continue

            cleaned.append(line)

        return "\n".join(cleaned)

    def run(self):
        for file_name in os.listdir(self.input_dir):
            if not file_name.lower().endswith(".pdf"):
                continue

            pdf_path = os.path.join(self.input_dir, file_name)
            text = self.extract_text(pdf_path)
            text = self.clean_text(text)

            output_name = file_name.replace(".pdf", ".txt")
            output_path = os.path.join(self.output_dir, output_name)

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(text)

            print(f"Extracted: {output_path}")