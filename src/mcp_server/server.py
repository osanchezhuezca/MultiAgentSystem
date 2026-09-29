# src/mcp_server/server.py
from mcp.server.fastmcp import FastMCP
from src.agents.sql_agent import build_sql_agent
from src.agents.rag_agent import build_rag_agent

mcp = FastMCP("CustomerSupportAgents")

@mcp.tool()
async def query_customer_database(natural_language_query: str) -> str:
    """Query the customer SQL database using natural language.
    Use for questions about customer profiles, ticket history, priorities, and statuses."""
    agent = build_sql_agent("sqlite:///data/processed/customer_support.db")
    result = await agent.ainvoke({"input": natural_language_query})
    return result["output"]

@mcp.tool()
async def query_policy_documents(question: str) -> str:
    """Answer questions about company policies from uploaded PDF documents.
    Use for refund, returns, shipping, and service policy questions."""
    chain = build_rag_agent("data/processed/chroma_db")
    return await chain.ainvoke(question)

if __name__ == "__main__":
    mcp.run(transport="stdio")