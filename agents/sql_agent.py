import os
from langchain_groq import ChatGroq
from langchain_community.utilities import SQLDatabase
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from dotenv import load_dotenv

load_dotenv()

def load_sql_agent(db_path: str = "data/database/banking.db"):
    """Custom SQL chain using LCEL — more reliable than agent for Llama models."""

    db = SQLDatabase.from_uri(f"sqlite:///{db_path}")

    llm = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.3-70b-versatile",
        temperature=0
    )

    # Step 1 — Generate SQL from question
    sql_generation_prompt = PromptTemplate(
        template="""You are a SQL expert. Given the database schema and question, write a SQLite SQL query.
Return ONLY the SQL query, nothing else. No explanation, no markdown, no backticks.

Database Schema:
{schema}

Question: {question}

SQL Query:""",
        input_variables=["schema", "question"]
    )

    # Step 2 — Generate final answer from SQL result
    answer_prompt = PromptTemplate(
        template="""Given the question, SQL query, and result, write a clear answer.

Question: {question}
SQL Query: {query}
SQL Result: {result}

Answer:""",
        input_variables=["question", "query", "result"]
    )

    def run_sql_chain(question: str) -> str:
        try:
            # Get schema
            schema = db.get_table_info()

            # Generate SQL
            sql_chain = sql_generation_prompt | llm | StrOutputParser()
            sql_query = sql_chain.invoke({
                "schema": schema,
                "question": question
            }).strip()

            # Clean up query if needed
            sql_query = sql_query.replace("```sql", "").replace("```", "").strip()

            # Execute SQL
            result = db.run(sql_query)

            # Generate answer
            answer_chain = answer_prompt | llm | StrOutputParser()
            answer = answer_chain.invoke({
                "question": question,
                "query": sql_query,
                "result": result
            })

            return answer

        except Exception as e:
            return f"I encountered an error processing your query: {str(e)}"

    # Wrap as a dict-compatible interface to match orchestrator
    class SQLChainWrapper:
        def invoke(self, input_dict):
            question = input_dict.get("input", "")
            output = run_sql_chain(question)
            return {"output": output}

    return SQLChainWrapper()