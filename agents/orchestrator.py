from langgraph.graph import StateGraph, END
from typing import TypedDict
from guardrails.input_guardrail import run_input_guardrail

class AgentState(TypedDict):
    query: str
    agent_used: str
    response: str
    sources: list[str]

def build_orchestrator(rag_chain, sql_agent):
    """Build the LangGraph orchestrator that routes between RAG and SQL agents."""

    # Input guardrail node
    def input_guardrail(state: AgentState) -> AgentState:
        result = run_input_guardrail(state["query"])

        if result["blocked"]:
            state["agent_used"] = "guardrail"
            state["response"] = result["response"]
            state["sources"] = []

        return state

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
        rag_result = rag_chain(state["query"])
        state["response"] = rag_result["answer"]
        state["sources"] = rag_result["sources"]
        return state

    # SQL node
    def run_sql_agent(state: AgentState) -> AgentState:
        result = sql_agent.invoke({"input": state["query"]})
        state["response"] = result["output"]
        state["sources"] = [] 
        return state

    # Routing function
    def route_to_agent(state: AgentState) -> str:
        return state["agent_used"]

    # Guardrail routing function
    def route_after_guardrail(state: AgentState) -> str:
        return "blocked" if state["agent_used"] == "guardrail" else "router"

    # Build graph
    workflow = StateGraph(AgentState)
    workflow.add_node("input_guardrail", input_guardrail)
    workflow.add_node("router", router)
    workflow.add_node("rag", run_rag_agent)
    workflow.add_node("sql", run_sql_agent)
    workflow.set_entry_point("input_guardrail")
    workflow.add_conditional_edges(
        "input_guardrail",
        route_after_guardrail,
        {"blocked": END, "router": "router"}
    )
    workflow.add_conditional_edges(
        "router", route_to_agent, {"rag": "rag", "sql": "sql"}
    )
    workflow.add_edge("rag", END)
    workflow.add_edge("sql", END)

    return workflow.compile()
