"""SQL agent — queries structured customer data via LangGraph react agent."""
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

from src.config import settings
from src.llm import get_llm

SQL_SYSTEM_PROMPT = """You are a customer support assistant with access to a SQL
database containing customer profiles and support tickets.

Workflow:
1. Use the `sql_db_list_tables` tool to see available tables.
2. Use `sql_db_schema` to inspect the schema of tables you need.
3. Write a query and run it with `sql_db_query`.
4. If the query fails, read the error and correct it.
5. Return a concise, human-friendly answer — do not dump raw SQL to the user.

Rules:
- Only run SELECT statements. Never INSERT, UPDATE, DELETE, or DROP.
- Always LIMIT results to 20 rows unless the user asks for a count.
- If the answer is a number (count, average), state it plainly.
"""


def build_sql_agent(db_url: str | None = None, llm=None):
    """Return a LangGraph react agent wired to the customer SQL database."""
    db_url = db_url or settings.database_url
    db = SQLDatabase.from_uri(db_url)
    llm = llm or get_llm()
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    tools = toolkit.get_tools()
    agent = create_react_agent(llm, tools, prompt=SQL_SYSTEM_PROMPT)
    return agent


def run_sql_query(query: str, db_url: str | None = None) -> str:
    """Run a natural-language query and return just the final answer string."""
    agent = build_sql_agent(db_url)
    result = agent.invoke({"messages": [HumanMessage(content=query)]})
    final = result["messages"][-1]
    return final.content if hasattr(final, "content") else str(final)
