import os
import re
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# ── CONFIGURATION ──
DOCS_PATH = "data/documents/"
VECTORSTORE_PATH = "vectorstore/"

PDF_FILES = [
    "hdfc_credit_card_policy.pdf",
    "hdfc_customer_compensation_policy.pdf",
    "hdfc_grievance_policy.pdf",
    "hdfc_personal_loan_agreement.pdf",
    "hdfc_savings_account_charges.pdf",
    "hdfc_general_terms_conditions.pdf"
]

def clean_text(text: str) -> str:
    text = re.sub(r'Classification\s*[-–]\s*Internal', '', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'\s{3,}', ' ', text)
    text = re.sub(r'as on \d{2}\.\d{2}\.\d{4}', '', text)
    return text.strip()

def ingest():
    print(" Loading PDFs...")
    all_documents = []
    for pdf in PDF_FILES:
        path = os.path.join(DOCS_PATH, pdf)
        if not os.path.exists(path):
            print(f"  Skipping missing file: {pdf}")
            continue
        loader = PyPDFLoader(path)
        docs = loader.load()
        all_documents.extend(docs)
        print(f"   {pdf} — {len(docs)} pages")

    print(f"\n Total pages: {len(all_documents)}")

    # Split
    print("\n  Splitting into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = splitter.split_documents(all_documents)

    # Clean
    for chunk in chunks:
        chunk.page_content = clean_text(chunk.page_content)
    chunks = [c for c in chunks if len(c.page_content) > 50]
    print(f" Chunks after cleaning: {len(chunks)}")

    # Embed + Save FAISS
    print("\n Building FAISS index...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)

    os.makedirs(VECTORSTORE_PATH, exist_ok=True)
    vectorstore.save_local(VECTORSTORE_PATH)
    print(f" FAISS index saved to '{VECTORSTORE_PATH}'")
    print(f" Total vectors: {vectorstore.index.ntotal}")

if __name__ == "__main__":
    ingest()