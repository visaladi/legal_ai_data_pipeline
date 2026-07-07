# agents/metadata_agent.py

import os
import re
import json
from tqdm import tqdm


class MetadataAgent:
    def __init__(self, input_dir="data/extracted_text", output_dir="data/metadata"):
        self.input_dir = input_dir
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def extract_metadata(self, filename: str, text: str):
        metadata = {
            "file_name": filename,
            "case_number": None,
            "court": None,
            "decision_date": None,
            "judges": [],
            "legal_area": None,
            "parties": None
        }

        case_match = re.search(
            r"(SC\s?/?\s?(APPEAL|FR|CHC|SPL|TAB)[^\n]*)",
            text,
            re.IGNORECASE
        )
        if case_match:
            metadata["case_number"] = case_match.group(1).strip()

        if "SUPREME COURT" in text.upper():
            metadata["court"] = "Supreme Court of Sri Lanka"
        elif "COURT OF APPEAL" in text.upper():
            metadata["court"] = "Court of Appeal of Sri Lanka"

        date_match = re.search(
            r"(\d{1,2}[./-]\d{1,2}[./-]\d{2,4})",
            text
        )
        if date_match:
            metadata["decision_date"] = date_match.group(1)

        judge_matches = re.findall(
            r"([A-Z][A-Za-z .]+,\s?J\.?)",
            text
        )
        metadata["judges"] = list(set(judge_matches))

        return metadata

    def process_all_texts(self):
        all_metadata = []

        txt_files = [
            f for f in os.listdir(self.input_dir)
            if f.lower().endswith(".txt")
        ]

        for txt_file in tqdm(txt_files, desc="Extracting metadata"):
            path = os.path.join(self.input_dir, txt_file)

            with open(path, "r", encoding="utf-8") as f:
                text = f.read()

            metadata = self.extract_metadata(txt_file, text)
            all_metadata.append(metadata)

        output_path = os.path.join(self.output_dir, "metadata.json")

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_metadata, f, indent=4, ensure_ascii=False)

        return all_metadata