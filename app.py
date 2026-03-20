import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import streamlit as st

import streamlit as st
from agents.rag_agent import load_rag_agent
from agents.sql_agent import load_sql_agent
from agents.orchestrator import build_orchestrator

import re

def format_currency(text):
    return re.sub(r"\$(\d+(?:,\d+)*(?:\.\d+)?)", r"₹\1", text)

# ── PAGE CONFIG ──
st.set_page_config(
    page_title="HDFC Banking Intelligence Assistant",
    page_icon="🏦",
    layout="centered"
)

# ── HEADER ──
st.title(" HDFC Banking Intelligence Assistant")
st.markdown("""
Ask me anything about **HDFC Bank policies** or your **account & transaction data**.
I'll automatically route your question to the right agent.
""")
st.divider()

# ── LOAD AGENTS (cached so they load only once) ──
@st.cache_resource
def load_agents():
    with st.spinner("Loading agents... please wait "):
        rag_chain = load_rag_agent()
        sql_agent = load_sql_agent()
        orchestrator = build_orchestrator(rag_chain, sql_agent)
    return orchestrator

orchestrator = load_agents()

# ── CHAT HISTORY ──
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── SAMPLE QUESTIONS ──
if len(st.session_state.messages) == 0:
    st.markdown("#### Try asking:")
    col1, col2 = st.columns(2)
    with col1:
        st.info(" What is the minimum balance for a savings account?")
        st.info(" How can I raise a grievance against HDFC Bank?")
        st.info(" What are the KYC documents required?")
    with col2:
        st.info(" Which customers have overdue credit cards?")
        st.info(" Which merchant has the highest transactions?")
        st.info(" What is the average balance by account type?")

# ── CHAT INPUT ──
if query := st.chat_input("Ask your banking question here..."):

    # Add user message
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Get response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = orchestrator.invoke({
                "query": query,
                "agent_used": "",
                "response": ""
            })

        response = format_currency(result["response"])
        agent_used = result["agent_used"].upper()

        # Show which agent handled it
        if agent_used == "RAG":
            st.caption(" Answered by: Policy Agent (RAG)")
        else:
            st.caption(" Answered by: Data Agent (SQL)")

        if "Sources:" in response:
            answer, sources = response.split("Sources:")

        # Show answer
            st.markdown(answer)

        # Show sources nicely
            st.markdown("### Sources")
            for s in sources.strip().split("\n"):
                if s.strip():
                   st.markdown(f"- {s.replace('-', '').strip()}")
        else:
            st.markdown(response)

    if "Why this answer?" in response:
        explanation = response.split("Why this answer?")[1]

        st.markdown("### Why this answer?")
        st.markdown(explanation)

    # Save assistant message
    st.session_state.messages.append({
        "role": "assistant",
        "content": f"*[{agent_used} Agent]*\n\n{response}"
    })