from PyPDF2 import PdfReader
from io import BytesIO

def extract_text(pdf_bytes):
    reader = PdfReader(BytesIO(pdf_bytes))
    return "".join([(p.extract_text() or "") for p in reader.pages])
