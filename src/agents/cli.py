"""Command-line interface for testing the agents individually.

Usage:
    python -m src.agents.cli sql "How many open tickets are there?"
    python -m src.agents.cli rag "What does the terms document say?"
    python -m src.agents.cli supervisor "Show me Ema's tickets and the refund policy"
"""
from __future__ import annotations

import argparse


def run_sql(query: str) -> str:
    from src.agents.sql_agent import run_sql_query
    from src.config import settings
    return run_sql_query(query, settings.database_url)

def run_rag(query: str) -> str:
    from src.agents.rag_agent import build_rag_agent
    from src.config import settings
    chain = build_rag_agent(settings.chroma_persist_dir)
    return chain.invoke(query)


def run_supervisor(query: str) -> dict:
    from src.agents.supervisor import build_supervisor_graph
    graph = build_supervisor_graph()
    return graph.invoke({"query": query})


def main() -> None:
    parser = argparse.ArgumentParser(description="Agent CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    for name in ("sql", "rag", "supervisor"):
        p = sub.add_parser(name, help=f"Run the {name} agent")
        p.add_argument("query", help="Natural-language query")

    args = parser.parse_args()

    if args.command == "sql":
        print(run_sql(args.query))
    elif args.command == "rag":
        print(run_rag(args.query))
    elif args.command == "supervisor":
        result = run_supervisor(args.query)
        print(f"[route: {result.get('route')}]")
        print(result.get("final_answer", ""))


if __name__ == "__main__":
    main()
