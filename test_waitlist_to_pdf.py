
import os
from urllib.parse import urlparse
from test_playwright import webpage_to_pdf
from test_ragflow import exist_doc_in_ragflow, upload_to_ragflow

if __name__ == "__main__":
    with open("waiting_list.txt", "r", encoding="utf-8") as f:
        for line in f.readlines():
            url, output_path = line.strip().split(" -> ")
            if not exist_doc_in_ragflow("CRAWLER DATASET", output_path) and urlparse(url).path.split('.')[-1] not in ["docx", "md", "txt", "pdf", "vsdx"]:
                webpage_to_pdf(url, "temp.pdf", timeout=600000)
                upload_to_ragflow(output_path, "CRAWLER DATASET", "temp.pdf")
                os.remove("temp.pdf")