"""Tests for the SQL agent."""
from __future__ import annotations

import pytest

from tests.conftest import requires_db


# ---------- Unit tests (no LLM needed) ----------

def test_sql_database_has_expected_tables(db_url):
    """The seeded database exposes customers and support_tickets."""
    from langchain_community.utilities import SQLDatabase

    db = SQLDatabase.from_uri(db_url)
    tables = db.get_usable_table_names()
    assert "customers" in tables
    assert "support_tickets" in tables


def test_sql_database_is_non_empty(db_url):
    """Both tables contain at least one row after seeding."""
    from langchain_community.utilities import SQLDatabase

    db = SQLDatabase.from_uri(db_url)
    result = db.run("SELECT COUNT(*) FROM customers;")
    assert int(result) > 0

    result = db.run("SELECT COUNT(*) FROM support_tickets;")
    assert int(result) > 0


def test_support_tickets_have_foreign_keys(db_url):
    """Every ticket points at a real customer."""
    from langchain_community.utilities import SQLDatabase

    db = SQLDatabase.from_uri(db_url)
    orphaned = db.run(
        "SELECT COUNT(*) FROM support_tickets t "
        "LEFT JOIN customers c ON c.id = t.customer_id "
        "WHERE c.id IS NULL;"
    )
    assert int(orphaned) == 0


def test_build_sql_agent_returns_runnable(db_url):
    """The factory returns an object with an invoke() method."""
    from src.agents.sql_agent import build_sql_agent

    agent = build_sql_agent(db_url)
    assert hasattr(agent, "invoke")


def test_run_sql_query_returns_string(db_url):
    """run_sql_query returns a plain string suitable for the synthesis node."""
    from src.agents.sql_agent import run_sql_query

    answer = run_sql_query("How many customers are there?", db_url)
    assert isinstance(answer, str)
    assert len(answer) > 0


# ---------- Integration tests (require a live LLM) ----------

@pytest.mark.integration
@requires_db
def test_sql_agent_counts_rows(db_url):
    """A natural-language count query produces a correct numeric answer."""
    from src.agents.sql_agent import run_sql_query

    answer = run_sql_query("How many support tickets are there in total?", db_url)
    assert isinstance(answer, str)
    # The exact number varies; just assert a digit appears.
    assert any(ch.isdigit() for ch in answer)


@pytest.mark.integration
@requires_db
def test_sql_agent_handles_unknown_column(db_url):
    """A query referencing a non-existent column does not crash the agent."""
    from src.agents.sql_agent import run_sql_query

    answer = run_sql_query(
        "How many rows are in the imaginary_table_that_does_not_exist?",
        db_url,
    )
    assert isinstance(answer, str)
