from langgraph.graph import StateGraph, END
from typing import TypedDict

class AgentState(TypedDict):
    query: str
    agent_used: str
    response: str

def build_orchestrator(rag_chain, sql_agent):
    """Build the LangGraph orchestrator that routes between RAG and SQL agents."""

    # Router node
    def router(state: AgentState) -> AgentState:
        query = state["query"].lower()

        sql_keywords = [
            "transaction", "balance", "how many", "outstanding",
            "credit card", "merchant", "customer", "branch",
            "average", "total", "count", "highest", "lowest",
            "failed", "blocked", "overdue", "statement"
        ]
        rag_keywords = [
            "policy", "rule", "guideline", "what is", "how does",
            "eligibility", "penalty", "interest rate", "fee",
            "grievance", "kyc", "document", "complaint", "process",
            "minimum balance", "loan", "terms", "conditions"
        ]

        sql_score = sum(1 for kw in sql_keywords if kw in query)
        rag_score = sum(1 for kw in rag_keywords if kw in query)
        state["agent_used"] = "sql" if sql_score > rag_score else "rag"
        return state

    # RAG node
    def run_rag_agent(state: AgentState) -> AgentState:
        state["response"] = rag_chain.invoke(state["query"])
        return state

    # SQL node
    def run_sql_agent(state: AgentState) -> AgentState:
        result = sql_agent.invoke({"input": state["query"]})
        state["response"] = result["output"]
        return state

    # Routing function
    def route_to_agent(state: AgentState) -> str:
        return state["agent_used"]

    # Build graph
    workflow = StateGraph(AgentState)
    workflow.add_node("router", router)
    workflow.add_node("rag", run_rag_agent)
    workflow.add_node("sql", run_sql_agent)
    workflow.set_entry_point("router")
    workflow.add_conditional_edges(
        "router", route_to_agent, {"rag": "rag", "sql": "sql"}
    )
    workflow.add_edge("rag", END)
    workflow.add_edge("sql", END)

    return workflow.compile()