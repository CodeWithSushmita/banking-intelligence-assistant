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

import os
import requests

def load_documents():
    docs = []

    base_url = "https://huggingface.co/datasets/MLbySush/banking-rag-documents/resolve/main"

    files = [
        "hdfc_credit_card_policy.pdf",
        "hdfc_customer_compensation_policy.pdf",
        "hdfc_general_terms_conditions.pdf",
        "hdfc_grievance_policy.pdf",
        "hdfc_personal_loan_agreement.pdf",
        "hdfc_savings_account_charges.pdf",
    ]

    for file in files:
        try:
            url = f"{base_url}/{file}"
            loader = PyPDFLoader(url)
            docs.extend(loader.load())
        except Exception as e:
            print(f"Error loading {file}: {e}")

    return docs

def load_rag_agent(vectorstore_path: str = "vectorstore/"):
    """Load the RAG agent from saved FAISS vectorstore."""

    # Load embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Load FAISS index
    if os.path.exists("vectorstore/index.faiss"):
        vectorstore = FAISS.load_local(
            "vectorstore",
             embeddings,
             allow_dangerous_deserialization=True
    )
    else:
        documents = load_documents()   #PDF loader

        if not documents:
            raise ValueError("No documents loaded. Check dataset URLs.")
        
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

    STRICT RULES:
    - Do NOT generate or mention any "Sources"
    - Do NOT say "based on the document"
    - Only answer from context
    - If answer is not found, say you don't have enough information

    Context:
    {context}

    Customer Question: {question}

    Answer:"""

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    def extract_sources(docs):
        sources = []

        for doc in docs:
            source = doc.metadata.get("source", "Unknown")
            filename = os.path.basename(source)
            sources.append(filename)

        return list(set(sources))

    def run_rag(question):
        # 1. Retrieve documents
        docs = retriever.invoke(question)

        # 2. Create context for LLM
        context = "\n\n".join(doc.page_content for doc in docs)

        # 3. Extract sources separately
        sources = []
        for doc in docs:
            source = doc.metadata.get("source", "Unknown")
            filename = os.path.basename(source)
            sources.append(filename)

        sources = list(set(sources))

        # 4. Call LLM
        response = llm.invoke(
            prompt.format(context=context, question=question)
        )

        return {
            "answer": response.content,
            "sources": sources
        }

    return run_rag