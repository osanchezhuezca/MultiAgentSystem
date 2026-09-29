# src/agents/supervisor.py
from typing import Literal, TypedDict, Annotated
import operator
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from src.llm import get_llm, get_router_llm
from pydantic import BaseModel, Field

class RouterDecision(BaseModel):
    route: Literal["sql", "rag", "both"] = Field(
        description="Which agent(s) should handle this query"
    )
    sql_query: str = Field(default="", description="Reformulated query for SQL agent")
    rag_query: str = Field(default="", description="Reformulated query for RAG agent")

class AgentState(TypedDict):
    query: str
    route: str
    sql_query: str        
    rag_query: str         
    sql_result: Annotated[str, operator.add]
    rag_result: Annotated[str, operator.add]
    final_answer: str

def router_node(state: AgentState):
    llm = get_router_llm()     
    classifier = llm.with_structured_output(RouterDecision)
    decision = classifier.invoke(
        f"Classify this customer support query. Route to 'sql' for customer data "
        f"or ticket lookups, 'rag' for policy/policy document questions, or 'both' "
        f"if it requires both. Query: {state['query']}"
    )
    return {
        "route": decision.route,
        "sql_query": decision.sql_query or state["query"],
        "rag_query": decision.rag_query or state["query"]
    }

def route_decision(state: AgentState):
    route = state.get("route") or "rag"
    query = state.get("query") or ""
    sql_query = state.get("sql_query") or query
    rag_query = state.get("rag_query") or query

    if route == "both":
        return [
            Send("sql_agent", {"query": sql_query}),
            Send("rag_agent", {"query": rag_query}),
        ]
    if route == "sql":
        return [Send("sql_agent", {"query": sql_query})]
    return [Send("rag_agent", {"query": rag_query})]

def sql_agent_node(state: dict):
    from src.agents.sql_agent import run_sql_query
    from src.config import settings
    result = run_sql_query(state["query"], settings.database_url)
    return {"sql_result": result}

def rag_agent_node(state: AgentState):
    from .rag_agent import build_rag_agent
    chain = build_rag_agent("data/processed/chroma_db")
    result = chain.invoke(state["query"])
    return {"rag_result": result}

def synthesis_node(state: AgentState):
    llm = get_llm()
    parts = []
    if state.get("sql_result"):
        parts.append(f"Customer Data:\n{state['sql_result']}")
    if state.get("rag_result"):
        parts.append(f"Policy Information:\n{state['rag_result']}")
    
    answer = llm.invoke(
        f"Synthesize a single, coherent answer for the customer support agent "
        f"based on these results:\n\n" + "\n\n".join(parts)
    )
    return {"final_answer": answer.content}

def build_supervisor_graph():
    graph = StateGraph(AgentState)
    graph.add_node("router", router_node)
    graph.add_node("sql_agent", sql_agent_node)
    graph.add_node("rag_agent", rag_agent_node)
    graph.add_node("synthesis", synthesis_node)
    
    graph.add_edge(START, "router")
    graph.add_conditional_edges("router", route_decision, 
                                ["sql_agent", "rag_agent"])
    graph.add_edge("sql_agent", "synthesis")
    graph.add_edge("rag_agent", "synthesis")
    graph.add_edge("synthesis", END)
    
    return graph.compile()