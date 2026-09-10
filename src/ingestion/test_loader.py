from loader import load_pdf

pdf_path = "data/raw/rag_original.pdf"

text = load_pdf(pdf_path)

print("Characters extracted:", len(text))
print()
print(text[:2000])