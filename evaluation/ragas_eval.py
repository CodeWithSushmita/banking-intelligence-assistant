"""
Standalone RAGAS evaluation pipeline for the Banking Intelligence Assistant.

This script is intentionally kept outside the Streamlit app and LangGraph
orchestration. It mirrors the current production RAG setup, runs an offline
evaluation over a CSV test set, and writes detailed results to CSV.

Usage:
    venv/bin/python evaluation/ragas_eval.py

Required extra dependencies for evaluation:
    pip install ragas
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import pandas as pd
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from ragas import evaluate
from ragas.dataset_schema import EvaluationDataset
from ragas.metrics._answer_relevance import AnswerRelevancy
from ragas.metrics._faithfulness import Faithfulness

from langchain_huggingface import HuggingFaceEmbeddings


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVALUATION_DIR = PROJECT_ROOT / "evaluation"
TEST_QUESTIONS_PATH = EVALUATION_DIR / "test_questions.csv"
RESULTS_PATH = EVALUATION_DIR / "results.csv"
VECTORSTORE_PATH = PROJECT_ROOT / "vectorstore"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL_NAME = "llama-3.1-8b-instant"
RETRIEVER_SEARCH_TYPE = "mmr"
RETRIEVER_K = 4
RETRIEVER_FETCH_K = 20
RETRIEVER_LAMBDA_MULT = 0.7


PROMPT_TEMPLATE = """You are a helpful HDFC Bank policy assistant.

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


def load_rag_components():
    """
    Load the same embedding model, FAISS index, retriever settings, and Groq
    model used by the current production RAG branch.
    """

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    vectorstore = FAISS.load_local(
        str(VECTORSTORE_PATH),
        embeddings,
        allow_dangerous_deserialization=True,
    )
    retriever = vectorstore.as_retriever(
        search_type=RETRIEVER_SEARCH_TYPE,
        search_kwargs={
            "k": RETRIEVER_K,
            "fetch_k": RETRIEVER_FETCH_K,
            "lambda_mult": RETRIEVER_LAMBDA_MULT,
        },
    )
    llm = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model_name=GROQ_MODEL_NAME,
        temperature=0,
    )
    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"],
    )
    return embeddings, retriever, llm, prompt


def build_eval_models(langchain_llm: ChatGroq, langchain_embeddings: HuggingFaceEmbeddings):
    """
    Return the evaluation models in the format expected by ragas 0.4.3.

    The deprecated evaluate() path in ragas 0.4.3 still validates metrics
    against the legacy Metric base class and can automatically wrap LangChain
    LLMs and embeddings internally. To stay compatible, we keep using the same
    ChatGroq and HuggingFaceEmbeddings instances from the current RAG setup.
    """

    return langchain_llm, langchain_embeddings


def extract_sources(docs: Iterable) -> list[str]:
    """
    Extract unique source filenames from retrieved documents while preserving
    their original retrieval order.
    """

    ordered_sources: list[str] = []
    seen = set()
    for doc in docs:
        source = Path(doc.metadata.get("source", "Unknown")).name
        if source not in seen:
            ordered_sources.append(source)
            seen.add(source)
    return ordered_sources


def run_rag_for_question(question: str, retriever, llm: ChatGroq, prompt: PromptTemplate) -> dict:
    """
    Run the same retrieve-then-generate flow as production, while also
    retaining the retrieved chunk texts needed by RAGAS.
    """

    docs = retriever.invoke(question)
    retrieved_contexts = [doc.page_content for doc in docs]
    context = "\n\n".join(retrieved_contexts)
    response = llm.invoke(prompt.format(context=context, question=question))

    return {
        "answer": response.content,
        "retrieved_contexts": retrieved_contexts,
        "sources": extract_sources(docs),
    }


def load_test_questions() -> pd.DataFrame:
    """
    Load the evaluation dataset from CSV and validate the expected columns.
    """

    df = pd.read_csv(TEST_QUESTIONS_PATH)
    required_columns = {"question", "ground_truth", "category"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required columns in test_questions.csv: {missing}")
    return df


def build_ragas_dataset(records_df: pd.DataFrame) -> EvaluationDataset:
    """
    Build a RAGAS-compatible dataset for ragas 0.4.3.

    RAGAS metrics in 0.4.3 expect the following fields:
    - user_input
    - retrieved_contexts
    - response
    - reference

    Category and source metadata are intentionally excluded here because
    EvaluationDataset only preserves RAGAS sample fields. We merge that
    metadata back into the final CSV after scoring.
    """

    ragas_df = records_df[
        ["question", "ground_truth", "answer", "retrieved_contexts"]
    ].rename(
        columns={
            "question": "user_input",
            "ground_truth": "reference",
            "answer": "response",
        }
    )
    return EvaluationDataset.from_pandas(ragas_df)


def evaluate_records(records_df: pd.DataFrame, ragas_llm, ragas_embeddings) -> pd.DataFrame:
    """
    Run RAGAS metrics and return a row-level results DataFrame.
    """

    ragas_dataset = build_ragas_dataset(records_df)
    metrics = [Faithfulness(), AnswerRelevancy()]
    result = evaluate(
        dataset=ragas_dataset,
        metrics=metrics,
        llm=ragas_llm,
        embeddings=ragas_embeddings,
    )
    metric_df = result.to_pandas()[["faithfulness", "answer_relevancy"]]
    return pd.concat([records_df.reset_index(drop=True), metric_df], axis=1)


def print_summary(results_df: pd.DataFrame) -> None:
    """
    Print overall and category-wise summaries for quick offline inspection.
    """

    avg_faithfulness = results_df["faithfulness"].mean()
    avg_answer_relevancy = results_df["answer_relevancy"].mean()
    question_count = len(results_df)

    print("\nRAGAS Evaluation Summary")
    print(f"Average Faithfulness: {avg_faithfulness:.4f}")
    print(f"Average Answer Relevancy: {avg_answer_relevancy:.4f}")
    print(f"Number of Questions Evaluated: {question_count}")

    print("\nScores by Category")
    grouped = (
        results_df.groupby("category")[["faithfulness", "answer_relevancy"]]
        .mean()
        .sort_index()
    )
    print(grouped.to_string(float_format=lambda value: f"{value:.4f}"))


def main() -> None:
    """
    Execute the full standalone evaluation pipeline.
    """

    load_dotenv(PROJECT_ROOT / ".env")

    if not os.getenv("GROQ_API_KEY"):
        raise EnvironmentError("GROQ_API_KEY is not set. Add it to your environment or .env file.")

    langchain_embeddings, retriever, langchain_llm, prompt = load_rag_components()
    ragas_llm, ragas_embeddings = build_eval_models(langchain_llm, langchain_embeddings)

    questions_df = load_test_questions()
    records: list[dict] = []

    for row in questions_df.itertuples(index=False):
        rag_result = run_rag_for_question(
            question=row.question,
            retriever=retriever,
            llm=langchain_llm,
            prompt=prompt,
        )
        records.append(
            {
                "question": row.question,
                "ground_truth": row.ground_truth,
                "category": row.category,
                "answer": rag_result["answer"],
                "retrieved_contexts": rag_result["retrieved_contexts"],
                "sources": rag_result["sources"],
            }
        )

    records_df = pd.DataFrame(records)
    results_df = evaluate_records(records_df, ragas_llm, ragas_embeddings)
    results_df.to_csv(RESULTS_PATH, index=False)
    print_summary(results_df)


if __name__ == "__main__":
    main()
