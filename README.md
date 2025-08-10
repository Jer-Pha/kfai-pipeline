# KF-AI

An end-to-end ETL pipeline and RAG system for searching and analyzing Kinda Funny video content.

## Overview

This project implements a full **ETL (Extract, Transform, Load)** pipeline to build a specialized **Retrieval-Augmented Generation (RAG)** system from the ground up.

The ETL pipeline is responsible for:
1.  **Extracting** video metadata from a private database and raw transcript data from the YouTube API.
2.  **Transforming** the unstructured transcripts into clean, validated, and structured data using Large Language Models for correction and analysis.
3.  **Loading** the processed data and its vector embeddings into a PostgreSQL database powered by the `pgvector` extension.

This curated database serves as the knowledge base for the RAG system, which provides an accurate and context-aware question-answering experience. When a user asks a question, the system retrieves relevant transcript chunks and uses them to augment the context provided to a local LLM, which then generates a detailed, source-cited answer.

## Features

-   **End-to-End ETL Pipeline:** Manages the entire data lifecycle from raw, disparate sources to a clean, queryable vector database.
-   **Retrieval-Augmented Generation (RAG):** Implements a modern RAG architecture for accurate, verifiable question-answering, minimizing LLM hallucinations.
-   **LLM-Powered Data Cleaning:** Utilizes local LLMs within the transformation step to intelligently clean and validate unstructured text data.
-   **Source-Cited Answers:** The query agent identifies and presents the specific video and timestamp from which information was drawn.
-   **Professional Tooling:** Built with a focus on code quality, including 100% strict-mode type checking with Mypy and a comprehensive test suite using Pytest.
## Technology Stack

This project leverages a modern stack of open-source tools to build a robust ETL and AI querying pipeline. The key components are organized by their function within the application.

### Core AI & Vector Search

-   **LLM Orchestration:** [**LangChain**](https://www.langchain.com/) is used as the central framework to chain together prompts, models, and retrieval systems.
-   **Local LLM Inference:** [**Ollama**](https://ollama.com/) serves the large language models locally for all text cleaning, analysis, and question-answering tasks.
-   **Embeddings:** [**Hugging Face `sentence-transformers`**](https://huggingface.co/sentence-transformers) are used to generate the high-quality vector embeddings necessary for semantic search.
-   **Vector Database:** [**PostgreSQL**](https://www.postgresql.org/) with the [**`pgvector`**](https://github.com/pgvector/pgvector) extension acts as the primary data store, holding both the processed metadata and the vector embeddings.
-   **Deep Learning Backend:** [**PyTorch**](https://pytorch.org/) provides the foundational tensor computation for the Hugging Face embedding models.

### Data Pipeline & Database Management

-   **Database Interaction:**
    -   [**SQLAlchemy**](https://www.sqlalchemy.org/) provides a robust Object-Relational Mapper (ORM) for programmatic interaction with the database.
    -   [**Psycopg**](https://www.psycopg.org/psycopg3/) is the high-performance PostgreSQL database adapter.
    -   [**MySQL Connector**](https://dev.mysql.com/doc/connector-python/en/) is used in the initial extraction phase to connect to the source metadata database.

### Code Quality & Testing

-   **Testing Framework:** [**Pytest**](https://pytest.org/) is used for writing clean, scalable tests. Code coverage is measured with [**`pytest-cov`**](https://pytest-cov.readthedocs.io/en/latest/).
-   **Static Type Checking:** [**Mypy**](http://mypy-lang.org/) is configured to run in **`--strict`** mode, ensuring a high degree of type safety across the entire codebase.
-   **Environment Configuration:** [**`python-dotenv`**](https://github.com/theskumar/python-dotenv) manages environment variables for database credentials and API keys.

## Project Structure

The project's architecture is explicitly designed around the ETL pattern, ensuring a clear separation of concerns.

```
kf-ai/
├── src/kfai/          # Main installable package
│   ├── core/          # Core utilities (paths, types)
│   ├── extractors/    # Data extraction modules
│   ├── transformers/  # Transcript cleaning and validation
│   └── loaders/       # Vector store and query agent logic
└── tests/             # Pytest suite with mirrored structure
```

## Usage

The project is managed via a simple command-line interface. It's designed to be run as an installed package, which registers a clean `kfai` command as the main entry point.

```bash
$ kfai
```

This launches the interactive menu for accessing the various pipeline stages:

```text
--- Welcome to KFAI ---

What would you like to do?
    1. Load raw data from KFDB
    2. Process failed video IDs
    3. Clean raw transcripts
    4. Update the vector store
    5. Interact with the Query Agent

Enter the number of your choice (or 'q' to quit):
```

## Note on Data Sources

Please be aware that the data extraction step (`extractors/`) is configured to connect to a private database. Therefore, the full pipeline is not runnable out-of-the-box. The primary purpose of this public repository is to serve as a portfolio piece demonstrating modern Python application structure, testing, and AI pipeline implementation.
