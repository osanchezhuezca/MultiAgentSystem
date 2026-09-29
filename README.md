# Multi-Agent Customer Support System

A Generative AI powered multi-agent system that lets customer support teams query structured customer data and unstructured policy documents through a single natural language interface.

Built as a technical assessment for Tata Consultancy Services.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Technology Stack](#technology-stack)
- [Repository Structure](#repository-structure)
- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [Configuration](#configuration)
- [Data Preparation](#data-preparation)
- [Usage](#usage)
- [Example Queries](#example-queries)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)
- [Design Decisions](#design-decisions)

---

## Overview

Customer support executives often need to pull information from multiple systems to answer a single question. A refund inquiry might require checking a policy PDF, while a customer history question requires querying a CRM database. Switching between tools costs time and reduces answer quality.

This project solves that with a supervisor-based multi-agent architecture. A single natural language interface routes each query to the appropriate specialist agent:

- **SQL Agent** answers questions about customer profiles and support ticket history by generating and executing SQL against a relational database.
- **RAG Agent** answers questions about company policy by retrieving relevant passages from embedded PDF documents and grounding its response in that context.
- **Supervisor** classifies the incoming query, dispatches to one or both agents (in parallel when needed), and synthesizes a unified answer.

The system is exposed through three interfaces: a Streamlit chat application, a command line tool, and an MCP server for integration with external agent hosts.

---

## Features

- Natural language querying of structured customer and ticket data
- Retrieval-augmented question answering over policy PDF documents
- Parallel multi-agent execution for queries that span both data sources
- Runtime PDF upload and ingestion through the UI
- Pluggable LLM provider (OpenAI, Groq, or local Ollama)
- Pluggable embedding model
- Provider-agnostic configuration driven entirely by environment variables
- MCP server exposing both agents as tools

---

## Architecture

The system uses a supervisor (router) pattern implemented with LangGraph. The graph has four nodes and executes as follows:

```
                     +-----------------+
                     |   User Query    |
                     +--------+--------+
                              |
                              v
                     +--------+--------+
                     |     Router      |
                     | (classifies and |
                     |  rewrites query)|
                     +--------+--------+
                              |
            +-----------------+-----------------+
            |                 |                 |
            v                 v                 v
      +-----------+    +-----------+    +-----------+
      | SQL Agent |    | RAG Agent |    |   Both    |
      |           |    |           |    | (parallel)|
      +-----+-----+    +-----+-----+    +-----+-----+
            |                |                 |
            +--------+-------+---------+-------+
                     |                 |
                     v                 v
              +------+-----------------+------+
              |       Synthesis Node          |
              | (merges results into single   |
              |  coherent response)           |
              +---------------+---------------+
                              |
                              v
                     +--------+--------+
                     |  Final Answer   |
                     +-----------------+
```

### Component Responsibilities

**Router Node.** Uses a lightweight LLM with structured output to classify the query into one of three routes (`sql`, `rag`, or `both`) and optionally rewrite the query for each downstream agent.

**SQL Agent.** A LangGraph ReAct agent with access to SQLDatabaseToolkit tools. It lists tables, inspects schema, writes a query, and executes it. Only SELECT statements are permitted.

**RAG Agent.** A retrieval chain that embeds the query, retrieves the top-k chunks from Chroma, and generates an answer strictly grounded in the retrieved context. Refuses to answer when the context is insufficient.

**Synthesis Node.** When two agents run in parallel, this node merges their outputs into a single coherent response using a stronger LLM.

### Data Flow

```
Policy PDFs  -> PyPDFLoader -> RecursiveCharacterTextSplitter -> Embeddings -> Chroma
Customer CSV -> pandas -> SQLAlchemy ORM -> SQLite
```

---

## Technology Stack

| Layer              | Component                                        |
| ------------------ | ------------------------------------------------ |
| Orchestration      | LangGraph                                        |
| Agent framework    | LangChain, LangGraph prebuilt ReAct              |
| LLM providers      | OpenAI, Groq, Ollama (configurable)              |
| Embeddings         | BAAI/bge-small-en-v1.5 via sentence-transformers |
| Structured storage | SQLite via SQLAlchemy                            |
| Vector storage     | Chroma                                           |
| PDF parsing        | pypdf via LangChain community loaders            |
| Text splitting     | langchain-text-splitters                         |
| UI                 | Streamlit                                        |
| Tool protocol      | Model Context Protocol via FastMCP               |
| Configuration      | pydantic-settings                                |
| Language           | Python 3.10 or newer                             |

---

## Repository Structure

```
MultiAgentSystem/
├── README.md
├── requirements.txt
├── requirements-dev.txt
├── Makefile
├── install.sh
├── .env.example
├── .gitignore
├── .streamlit/
│   └── config.toml
├── data/
│   ├── raw/
│   │   ├── customer_support_tickets.csv
│   │   └── policy_documents/
│   │       └── *.pdf
│   └── processed/
│       ├── customer_support.db
│       └── chroma_db/
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── llm.py
│   ├── database/
│   │   ├── __init__.py
│   │   ├── models.py
│   │   └── seed_data.py
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── pdf_loader.py
│   │   ├── vector_store.py
│   │   └── build.py
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── sql_agent.py
│   │   ├── rag_agent.py
│   │   ├── supervisor.py
│   │   └── cli.py
│   ├── mcp_server/
│   │   ├── __init__.py
│   │   └── server.py
│   └── ui/
│       ├── __init__.py
│       └── app.py
├── tests/
│   ├── __init__.py
│   ├── test_sql_agent.py
│   ├── test_rag_agent.py
│   └── test_supervisor.py
└── docs/
    └── architecture.md
```

---

## Prerequisites

- Python 3.10 or newer (3.11 recommended)
- pip and venv
- An API key from one of: OpenAI, Groq, or a local Ollama installation
- Approximately 1 GB of disk space for dependencies and models

---

## Setup

### Option A: Automated Setup

```bash
chmod +x install.sh
./install.sh --dev
```

The installer creates a virtual environment, installs dependencies, scaffolds directories, and copies `.env.example` to `.env`.

### Option B: Manual Setup

```bash
# Clone the repository
git clone https://github.com/<username>/MultiAgentSystem.git
cd MultiAgentSystem

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate          # macOS / Linux
# .\.venv\Scripts\Activate.ps1     # Windows PowerShell

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Optional: install development dependencies
pip install -r requirements-dev.txt
```

### Verify Installation

```bash
python -c "from src.llm import get_llm; print(type(get_llm()).__name__)"
```

A successful install prints `ChatOpenAI`, `ChatGroq`, or `ChatOllama` depending on your configured provider.

---

## Configuration

All configuration is read from a `.env` file at the project root. Copy the template and edit it:

```bash
cp .env.example .env
```

### Required Settings

```
LLM_PROVIDER=groq
GROQ_API_KEY=gsk_your_key_here
```

### Common Settings

| Variable                   | Description                                               | Default                                          |
| -------------------------- | --------------------------------------------------------- | ------------------------------------------------ |
| `LLM_PROVIDER`           | Which provider to use:`openai`, `groq`, or `ollama` | `openai`                                       |
| `OPENAI_API_KEY`         | OpenAI API key (required if provider is`openai`)        | empty                                            |
| `GROQ_API_KEY`           | Groq API key (required if provider is`groq`)            | empty                                            |
| `OPENAI_LLM_MODEL`       | Main model name for OpenAI                                | `gpt-4o`                                       |
| `OPENAI_ROUTER_MODEL`    | Router model for OpenAI                                   | `gpt-4o-mini`                                  |
| `GROQ_LLM_MODEL`         | Main model name for Groq                                  | `openai/gpt-oss-120b`                          |
| `GROQ_ROUTER_MODEL`      | Router model for Groq                                     | `openai/gpt-oss-20b`                           |
| `LLM_TEMPERATURE`        | Sampling temperature                                      | `0`                                            |
| `LLM_MAX_TOKENS`         | Max output tokens                                         | `1024`                                         |
| `EMBEDDING_MODEL`        | HuggingFace embedding model name                          | `BAAI/bge-small-en-v1.5`                       |
| `EMBEDDING_DEVICE`       | `cpu` or `cuda`                                       | `cpu`                                          |
| `DATABASE_URL`           | SQLAlchemy connection string                              | `sqlite:///data/processed/customer_support.db` |
| `CHROMA_PERSIST_DIR`     | Path to persist the Chroma store                          | `data/processed/chroma_db`                     |
| `CHROMA_COLLECTION_NAME` | Chroma collection name                                    | `policy_docs`                                  |
| `RAG_TOP_K`              | Number of chunks retrieved per query                      | `4`                                            |
| `PDF_CHUNK_SIZE`         | Characters per chunk                                      | `512`                                          |
| `PDF_CHUNK_OVERLAP`      | Overlap between chunks                                    | `64`                                           |
| `LOG_LEVEL`              | Python logging level                                      | `INFO`                                         |

### Switching Providers

Change one line in `.env` and restart:

```
LLM_PROVIDER=openai    # or groq or ollama
```

No code changes are required. The `get_llm()` factory in `src/llm.py` instantiates the correct client based on the provider setting.

---

## Data Preparation

### Structured Data

The system expects a CSV of customer support tickets at `data/raw/customer_support_tickets.csv`. Any CSV with the following columns works:

- `Ticket ID`
- `Customer Name`
- `Customer Email`
- `Customer Age`
- `Customer Gender`
- `Product Purchased`
- `Date of Purchase`
- `Ticket Type`
- `Ticket Subject`
- `Ticket Description`
- `Ticket Status`
- `Resolution`
- `Ticket Priority`
- `Ticket Channel`
- `First Response Time`
- `Time to Resolution`
- `Customer Satisfaction Rating`

A suitable dataset is available on Kaggle: search for "Customer Support Ticket Dataset" by suraj520.

Load it into SQLite:

```bash
python -m src.database.seed_data
```

This creates `data/processed/customer_support.db` with two tables: `customers` and `support_tickets`. The seed script is idempotent; re-running it rebuilds both tables from scratch.

### Unstructured Data

Place policy PDFs in `data/raw/policy_documents/`. Any company policy documents work; the system was tested with Walmart's public policy PDFs.

Ingest them into Chroma:

```bash
python -m src.ingestion.build
```

The first run downloads the embedding model (approximately 130 MB) to the HuggingFace cache. Subsequent runs reuse it.

To force a clean rebuild:

```bash
python -m src.ingestion.build --reset
```

### Verify Both Stores

```bash
python -c "
import sqlite3
con = sqlite3.connect('data/processed/customer_support.db')
for t in ['customers', 'support_tickets']:
    print(t, con.execute(f'SELECT COUNT(*) FROM {t}').fetchone()[0])
"

python -c "
import chromadb
client = chromadb.PersistentClient(path='data/processed/chroma_db')
print('vectors:', client.get_collection('policy_docs').count())
"
```

---

## Usage

### Streamlit UI

```bash
streamlit run src/ui/app.py
```

Opens at `http://localhost:8501`. The interface provides:

- A chat box for natural language queries
- A sidebar for uploading and ingesting new policy PDFs
- An expandable "Routing details" panel under each answer showing which agent handled the query and what each agent returned

### Command Line

Each agent can be invoked independently:

```bash
# SQL agent
python -m src.agents.cli sql "How many open high-priority tickets are there?"

# RAG agent
python -m src.agents.cli rag "What does the Walmart terms document cover?"

# Full supervisor with routing
python -m src.agents.cli supervisor "Show me a customer's profile and the refund policy"
```

The supervisor CLI prints the routing decision followed by the synthesized answer.

### MCP Server

Start the Model Context Protocol server:

```bash
python -m src.mcp_server.server
```

The server exposes two tools over stdio transport:

| Tool                        | Description                                                    |
| --------------------------- | -------------------------------------------------------------- |
| `query_customer_database` | Run a natural language query against the customer SQL database |
| `query_policy_documents`  | Answer a question from the embedded policy documents           |

An MCP-compatible client (Claude Desktop, Cursor, or any LangChain MCP adapter) can connect to this server and use the tools directly.

### Make Targets

Common tasks are wrapped in the Makefile:

```bash
make install-data    # full setup including data ingestion
make seed            # rebuild SQLite from CSV
make vector          # rebuild Chroma from PDFs
make run-ui          # launch Streamlit
make run-mcp         # launch MCP server
make test            # run pytest
make clean           # remove venv and generated data
```

---

## Example Queries

These queries exercise each route and demonstrate the multi-agent behavior.

### Policy Question (routes to RAG)

```
What is the current refund policy?
```

### Customer Data Question (routes to SQL)

```
How many open high-priority tickets are there?
```

```
List the top five customers by number of tickets.
```

```
Show me all tickets for customer Ema.
```

### Combined Question (routes to both in parallel)

```
Give me an overview of a customer's profile and the current refund policy.
```

### Follow-up After PDF Upload

1. Upload a new policy PDF through the sidebar
2. Click "Ingest PDF"
3. Ask a question about the newly ingested document

The RAG agent will retrieve from the updated vector store immediately.

---

## Testing

Run the full test suite:

```bash
pytest -q
```

Run a single test file:

```bash
pytest tests/test_sql_agent.py -v
```

Tests are organized by component:

| File                         | Coverage                                                   |
| ---------------------------- | ---------------------------------------------------------- |
| `tests/test_sql_agent.py`  | SQL generation, execution, error handling                  |
| `tests/test_rag_agent.py`  | Retrieval, grounded generation, refusal on missing context |
| `tests/test_supervisor.py` | Routing decisions for sql, rag, and both cases             |

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'src'`

Run commands from the project root, or ensure the project root is on `sys.path`. The Streamlit app handles this internally by inserting the project root before imports.

### `Missing credentials` when calling the LLM

The API key is not being read. Verify:

```bash
grep OPENAI_API_KEY .env       # or GROQ_API_KEY
python -c "from src.config import settings; print(bool(settings.openai_api_key))"
```

Common causes: quotes around the value, a leading space, or a `.env` file in the wrong directory.

### Model not found errors

LLM providers retire models regularly. If you see `The model X does not exist`, update the model name in `.env` to a currently supported one. For Groq, `openai/gpt-oss-120b` and `openai/gpt-oss-20b` are current production models.

### torchvision import warnings in Streamlit

Streamlit's file watcher scans every importable submodule, and many transformers model modules import torchvision optionally. These messages are harmless. Disable the watcher by setting `fileWatcherType = "none"` in `.streamlit/config.toml`.

### Vector store not finding uploaded documents

Streamlit caches the supervisor graph. After uploading a new PDF, refresh the browser page or restart the app to pick up the updated Chroma collection.

### Streamlit is using the wrong Python

Streamlit runs from whichever Python is first on PATH. Always activate the virtual environment before launching:

```bash
source .venv/bin/activate
which streamlit         # should point inside .venv/bin
streamlit run src/ui/app.py
```

---

## Design Decisions

**Why a supervisor pattern instead of a single agent with tools.** A single ReAct agent with both SQL and RAG tools works for simple cases but becomes unreliable when the query requires both sources. The supervisor makes routing explicit, allows parallel execution, and produces cleaner traces for debugging.

**Why LangGraph.** It provides a typed state machine, native parallel dispatch via `Send`, and first-class support for the supervisor pattern. It also composes cleanly with the newer `create_react_agent` for tool-using subagents.

**Why Chroma over other vector stores.** Chroma is embedded (no separate server), persists to a local directory, and integrates directly with LangChain. For a single-user demo this removes an entire infrastructure layer.

**Why SQLite.** The dataset is small, the queries are analytical, and SQLite requires zero configuration. The `DATABASE_URL` setting accepts a Postgres connection string if the deployment grows.

**Why pydantic-settings for configuration.** It validates types at startup, reads from `.env` automatically, and makes the provider swap a one-line change. It also fails loudly when a required field is missing.

**Why expose MCP.** Model Context Protocol is becoming the standard way to make tools available to agent hosts. Wrapping the two agents as MCP tools means the same logic can be consumed from Claude Desktop, Cursor, or any other MCP-compatible environment without duplicating code.

---

## License

See `LICENSE` in the repository root.
