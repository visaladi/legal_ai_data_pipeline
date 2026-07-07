# agents/data_collection_agent.py

import os
import csv
import time
import hashlib
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, unquote


class DataCollectionAgent:
    def __init__(
        self,
        output_dir="data/raw_pdfs",
        metadata_path="data/pdf_metadata.csv",
        delay=1.0
    ):
        self.output_dir = output_dir
        self.metadata_path = metadata_path
        self.delay = delay

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.metadata_path), exist_ok=True)

        self.headers = {
            "User-Agent": "SriLankaLegalResearchBot/1.0 Academic Research"
        }

    def get_pdf_links(self, index_url: str):
        response = requests.get(index_url, headers=self.headers, timeout=30)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        links = []

        for a in soup.find_all("a", href=True):
            href = a["href"]

            if href.lower().endswith(".pdf"):
                links.append(urljoin(index_url, href))

        return sorted(list(set(links)))

    def safe_filename(self, url: str):
        name = unquote(url.split("/")[-1])
        return name.replace("/", "_").replace("\\", "_")

    def file_hash(self, file_path):
        sha256 = hashlib.sha256()

        with open(file_path, "rb") as f:
            for block in iter(lambda: f.read(4096), b""):
                sha256.update(block)

        return sha256.hexdigest()

    def metadata_exists(self):
        return os.path.exists(self.metadata_path)

    def write_metadata(self, row):
        file_exists = self.metadata_exists()

        with open(self.metadata_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "source_url",
                    "file_name",
                    "local_path",
                    "file_size_bytes",
                    "sha256"
                ]
            )

            if not file_exists:
                writer.writeheader()

            writer.writerow(row)

    def download_pdf(self, url: str):
        file_name = self.safe_filename(url)
        file_path = os.path.join(self.output_dir, file_name)

        if os.path.exists(file_path):
            return file_path

        response = requests.get(url, headers=self.headers, timeout=90)
        response.raise_for_status()

        content_type = response.headers.get("Content-Type", "").lower()

        if "pdf" not in content_type and not url.lower().endswith(".pdf"):
            print(f"Skipped non-PDF: {url}")
            return None

        with open(file_path, "wb") as f:
            f.write(response.content)

        sha256 = self.file_hash(file_path)

        self.write_metadata({
            "source_url": url,
            "file_name": file_name,
            "local_path": file_path,
            "file_size_bytes": os.path.getsize(file_path),
            "sha256": sha256
        })

        time.sleep(self.delay)
        return file_path

    def run(self, index_url: str, limit=None):
        pdf_links = self.get_pdf_links(index_url)

        if limit is not None:
            pdf_links = pdf_links[:limit]

        print(f"Found {len(pdf_links)} PDFs")

        downloaded = []

        for i, url in enumerate(pdf_links, start=1):
            try:
                print(f"[{i}/{len(pdf_links)}] {url}")
                path = self.download_pdf(url)

                if path:
                    downloaded.append(path)

            except Exception as e:
                print(f"Failed: {url} | {e}")

        return downloaded