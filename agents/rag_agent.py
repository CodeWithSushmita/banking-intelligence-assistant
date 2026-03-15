import os
import requests
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv

load_dotenv()

def download_pdfs():
    folder = "data/documents"

    # Create folder if missing
    os.makedirs(folder, exist_ok=True)

    urls = {
        "hdfc_credit_card_policy.pdf": "https://github.com/CodeWithSushmita/banking-intelligence-assistant/blob/main/data/documents/hdfc_credit_card_policy.pdf",
        "hdfc_customer_compensation_policy.pdf": "https://github.com/CodeWithSushmita/banking-intelligence-assistant/blob/main/data/documents/hdfc_customer_compensation_policy.pdf",
        "hdfc_general_terms_conditions.pdf": "https://github.com/CodeWithSushmita/banking-intelligence-assistant/blob/main/data/documents/hdfc_general_terms_conditions.pdf",
        "hdfc_grievance_policy.pdf": "https://github.com/CodeWithSushmita/banking-intelligence-assistant/blob/main/data/documents/hdfc_grievance_policy.pdf",
        "hdfc_personal_loan_agreement.pdf": "https://github.com/CodeWithSushmita/banking-intelligence-assistant/blob/main/data/documents/hdfc_personal_loan_agreement.pdf",
        "hdfc_savings_account_charges.pdf": "https://github.com/CodeWithSushmita/banking-intelligence-assistant/blob/main/data/documents/hdfc_savings_account_charges.pdf",
    }

    for name, url in urls.items():
        path = f"data/documents/{name}"

        if not os.path.exists(path):
            r = requests.get(url)
            with open(path, "wb") as f:
                f.write(r.content)

def load_documents():
    docs = []
    folder = "data/documents"

    for file in os.listdir(folder):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(folder, file))
            docs.extend(loader.load())

    return docs

def load_rag_agent(vectorstore_path: str = "vectorstore/"):
    """Load the RAG agent from saved FAISS vectorstore."""

    download_pdfs()

    # Load embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Load FAISS index
    if os.path.exists("vectorstore/index.faiss"):
        vectorstore = FAISS.load_local("vectorstore", embeddings)
    else:
        documents = load_documents()   # your PDF loader
        vectorstore = FAISS.from_documents(documents, embeddings)
        vectorstore.save_local("vectorstore")


    # MMR retriever
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4, "fetch_k": 20, "lambda_mult": 0.7}
    )

    # LLM
    llm = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.1-8b-instant",
        temperature=0
    )

    # Grounded prompt
    prompt_template = """You are a helpful HDFC Bank policy assistant.
Use ONLY the context below to answer the customer's question.
If the answer is not in the context, say "I don't have enough information 
in the policy documents to answer this. Please contact HDFC Bank directly."

Context:
{context}

Customer Question: {question}

Answer:"""

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # LCEL chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain