from doc_segmentation import HierarchicalDocumentSegmenter
from dotenv import load_dotenv
import os
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
def process_pdf(pdf_path, output_path, api_key=None):
    segmenter = HierarchicalDocumentSegmenter(pdf_path, api_key)
    segments = segmenter.process_document()
    segmenter.export_to_json(output_path)
    return segments

if __name__ == "__main__":
    process_pdf("./RaghavResume.pdf", "./segments.json", groq_api_key)