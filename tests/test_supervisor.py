"""Tests for the supervisor router and graph."""
from __future__ import annotations

import pytest
from langgraph.types import Send

from tests.conftest import requires_chroma, requires_db


# ---------- Unit tests ----------

def test_route_decision_sql(empty_state):
    """A route of 'sql' dispatches to the SQL agent only."""
    from src.agents.supervisor import route_decision

    empty_state.update({
        "route": "sql",
        "query": "How many tickets?",
        "sql_query": "How many tickets?",
    })
    sends = route_decision(empty_state)

    assert isinstance(sends, list)
    assert len(sends) == 1
    assert isinstance(sends[0], Send)
    assert sends[0].node == "sql_agent"


def test_route_decision_rag(empty_state):
    """A route of 'rag' dispatches to the RAG agent only."""
    from src.agents.supervisor import route_decision

    empty_state.update({
        "route": "rag",
        "query": "What is the refund policy?",
        "rag_query": "What is the refund policy?",
    })
    sends = route_decision(empty_state)

    assert isinstance(sends, list)
    assert len(sends) == 1
    assert isinstance(sends[0], Send)
    assert sends[0].node == "rag_agent"


def test_route_decision_both(empty_state):
    """A route of 'both' dispatches to both agents in parallel."""
    from src.agents.supervisor import route_decision

    empty_state.update({
        "route": "both",
        "query": "Show me Ema's tickets and the refund policy",
        "sql_query": "Show me Ema's tickets",
        "rag_query": "What is the refund policy?",
    })
    sends = route_decision(empty_state)

    assert isinstance(sends, list)
    assert len(sends) == 2
    targets = {s.node for s in sends}
    assert targets == {"sql_agent", "rag_agent"}


def test_route_decision_defaults_to_rag(empty_state):
    """A missing route falls back to RAG instead of raising."""
    from src.agents.supervisor import route_decision

    empty_state.update({"query": "Anything"})
    sends = route_decision(empty_state)

    assert len(sends) == 1
    assert sends[0].node == "rag_agent"


def test_route_decision_handles_missing_specific_query(empty_state):
    """If sql_query is missing, the original query is used as a fallback."""
    from src.agents.supervisor import route_decision

    empty_state.update({"route": "sql", "query": "Fallback query"})
    sends = route_decision(empty_state)

    assert len(sends) == 1
    assert sends[0].node == "sql_agent"


def test_build_supervisor_graph_compiles():
    """The graph compiles without errors."""
    from src.agents.supervisor import build_supervisor_graph

    graph = build_supervisor_graph()
    # Compiled LangGraph objects expose .invoke and .stream
    assert hasattr(graph, "invoke")
    assert hasattr(graph, "stream")


def test_agent_state_has_required_fields():
    """AgentState declares every field the graph reads and writes."""
    from src.agents.supervisor import AgentState

    expected = {
        "query",
        "route",
        "sql_query",
        "rag_query",
        "sql_result",
        "rag_result",
        "final_answer",
    }
    assert expected.issubset(AgentState.__annotations__.keys())


# ---------- Integration tests ----------

@pytest.mark.integration
@requires_chroma
def test_supervisor_routes_policy_query_to_rag(empty_state):
    """A policy question is classified as a RAG query."""
    from src.agents.supervisor import build_supervisor_graph

    graph = build_supervisor_graph()
    empty_state["query"] = "What is the current refund policy?"
    result = graph.invoke(empty_state)

    assert result["route"] == "rag"
    assert len(result.get("final_answer", "")) > 0


@pytest.mark.integration
@requires_db
def test_supervisor_routes_customer_query_to_sql(empty_state):
    """A customer data question is classified as a SQL query."""
    from src.agents.supervisor import build_supervisor_graph

    graph = build_supervisor_graph()
    empty_state["query"] = "How many open high-priority tickets are there?"
    result = graph.invoke(empty_state)

    assert result["route"] == "sql"
    assert len(result.get("final_answer", "")) > 0


@pytest.mark.integration
@requires_db
@requires_chroma
def test_supervisor_combines_both_agents(empty_state):
    """A mixed query triggers both agents and produces a merged answer."""
    from src.agents.supervisor import build_supervisor_graph

    graph = build_supervisor_graph()
    empty_state["query"] = (
        "Show me a customer's ticket history and the current refund policy."
    )
    result = graph.invoke(empty_state)

    assert result["route"] == "both"
    assert len(result.get("sql_result", "")) > 0
    assert len(result.get("rag_result", "")) > 0
    assert len(result.get("final_answer", "")) > 0
