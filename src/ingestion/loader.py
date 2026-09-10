from pathlib import Path
from pypdf import PdfReader


def load_pdf(pdf_path: str) -> str:
    """
    Extract text from a PDF file.
    """

    path = Path(pdf_path)

    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    reader = PdfReader(path)

    pages = []

    for page in reader.pages:
        text = page.extract_text()

        if text:
            pages.append(text)

    return "\n".join(pages)