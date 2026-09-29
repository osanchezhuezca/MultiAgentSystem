"""Tests for the SQL agent."""
from __future__ import annotations

import sqlite3

import pytest

from tests.conftest import DB_PATH, requires_db


def _scalar(sql: str) -> int:
    """Run a single-value SQL query against the seeded database."""
    con = sqlite3.connect(str(DB_PATH))
    try:
        row = con.execute(sql).fetchone()
        return int(row[0])
    finally:
        con.close()


# ---------- Unit tests (no LLM needed) ----------

def test_sql_database_has_expected_tables():
    """The seeded database exposes customers and support_tickets."""
    from langchain_community.utilities import SQLDatabase
    from tests.conftest import DB_PATH as _DB

    db = SQLDatabase.from_uri(f"sqlite:///{_DB}")
    tables = db.get_usable_table_names()
    assert "customers" in tables
    assert "support_tickets" in tables


def test_sql_database_is_non_empty():
    """Both tables contain at least one row after seeding."""
    assert _scalar("SELECT COUNT(*) FROM customers") > 0
    assert _scalar("SELECT COUNT(*) FROM support_tickets") > 0


def test_support_tickets_have_foreign_keys():
    """Every ticket points at a real customer."""
    orphaned = _scalar(
        "SELECT COUNT(*) FROM support_tickets t "
        "LEFT JOIN customers c ON c.id = t.customer_id "
        "WHERE c.id IS NULL"
    )
    assert orphaned == 0


def test_priorities_are_populated():
    """Priority column has meaningful values the agent can filter on."""
    con = sqlite3.connect(str(DB_PATH))
    try:
        rows = con.execute(
            "SELECT DISTINCT priority FROM support_tickets "
            "WHERE priority IS NOT NULL"
        ).fetchall()
    finally:
        con.close()
    assert len(rows) > 0


def test_build_sql_agent_returns_runnable(db_url):
    """The factory returns an object with an invoke() method."""
    from src.agents.sql_agent import build_sql_agent

    agent = build_sql_agent(db_url)
    assert hasattr(agent, "invoke")


# ---------- Integration tests (require a live LLM) ----------

@pytest.mark.integration
@requires_db
def test_run_sql_query_returns_string(db_url):
    """run_sql_query returns a plain string suitable for the synthesis node."""
    from src.agents.sql_agent import run_sql_query

    answer = run_sql_query("How many customers are there?", db_url)
    assert isinstance(answer, str)
    assert len(answer) > 0


@pytest.mark.integration
@requires_db
@pytest.mark.slow
def test_sql_agent_counts_rows(db_url):
    """A natural-language count query produces a numeric answer."""
    from src.agents.sql_agent import run_sql_query

    answer = run_sql_query("How many support tickets are there in total?", db_url)
    assert isinstance(answer, str)
    assert any(ch.isdigit() for ch in answer)


@pytest.mark.integration
@requires_db
@pytest.mark.slow
def test_sql_agent_handles_unknown_column(db_url):
    """A query referencing a non-existent column does not crash the agent."""
    from src.agents.sql_agent import run_sql_query

    answer = run_sql_query(
        "How many rows are in the imaginary_table_that_does_not_exist?",
        db_url,
    )
    assert isinstance(answer, str)
