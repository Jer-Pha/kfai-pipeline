# KF/AI

[![Python Application CI](https://github.com/Jer-Pha/kf-ai/actions/workflows/ci.yml/badge.svg?branch=main&event=push)](https://github.com/Jer-Pha/kf-ai/actions/workflows/ci.yml)
[![Code style: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Checked with mypy](https://www.mypy-lang.org/static/mypy_badge.svg)](https://mypy-lang.org/)
[![Poetry](https://img.shields.io/endpoint?url=https://python-poetry.org/badge/v0.json)](https://python-poetry.org/)

An end-to-end ETL pipeline and RAG system for searching and analyzing Kinda Funny content.

## Overview

This project implements a full **ETL (Extract, Transform, Load)** pipeline to build a specialized **Retrieval-Augmented Generation (RAG)** system from the ground up.

A key feature of this system is its unique knowledge base, which is created by combining public data from the YouTube API with a rich, hand-curated set of metadata from the **[Kinda Funny Database (KFDB)](https://www.kfdb.app/) ([source](https://github.com/Jer-Pha/kfdb))**, a companion project that catalogs hosts, shows, and channel data.

The ETL pipeline is responsible for:

1.  **Extracting** video metadata from a private database and raw transcript data from the YouTube API.
2.  **Transforming** the unstructured transcripts into clean, validated, and structured data using Large Language Models for correction and analysis.
3.  **Loading** the processed data and its vector embeddings into a PostgreSQL database powered by the `pgvector` extension.

This curated database serves as the knowledge base for the RAG system, which provides an accurate and context-aware question-answering experience. When a user asks a question, the system retrieves relevant transcript chunks and uses them to augment the context provided to a local LLM, which then generates a detailed, source-cited answer.

## RAG in Action: An Example Query

The core of this project is its ability to answer complex, comparative questions by synthesizing information from across the entire Kinda Funny video library. Here is a real, unedited response from the query agent.

> **Question:**
> How did the crew's opinion of Cyberpunk 2077 change from its problematic launch to the release of the Phantom Liberty expansion?

> **Answer:**
> The crew's opinion of _Cyberpunk 2077_ evolved significantly from its problematic launch to the release of the _Phantom Liberty_ expansion, as reflected in multiple discussions across the _Kinda Funny_ archives. Initially, the game faced widespread criticism for technical issues, bugs, and unmet expectations, which likely shaped a negative perception among developers and players alike. However, the _Phantom Liberty_ expansion marked a turning point, with the crew acknowledging its role in "overhauling the way _Cyberpunk 2077_ is played." This expansion was praised for building on the game's foundation, addressing prior flaws, and delivering a "really good story" within the universe. The development team's efforts were highlighted, with _Phantom Liberty_ being the largest project at CD Projekt Red at the time, involving 300 people during its pre-release phase. Critics and developers alike noted that the expansion helped "get _Cyberpunk_ back on track," improving both sales and critical reception. Additionally, the expansion introduced new content such as a flying car, a revamped skill tree, and a new weapon, which were seen as significant enhancements. These changes, combined with the team's shift in focus from _Phantom Liberty_ to the next _The Witcher_ game (_Polaris_), signaled a renewed commitment to refining the _Cyberpunk_ experience. Overall, the _Phantom Liberty_ expansion appears to have mitigated the initial backlash, fostering a more positive outlook on the game's potential and the studio's ability to adapt.

> **Sources:**
>
> - **Video:** Official Summer Game Fest 2023 Predictions and Bets - Kinda Funny Gamescast
>   - **Timestamp:** `~34m 13s`
>   - **Link:** `https://www.youtube.com/watch?v=azbLcWwEC0Q&t=2053s`
> - **Video:** Discussing Starfield Reviews - Kinda Funny Games Daily 08.31.23
>   - **Timestamp:** `~55m 22s`
>   - **Link:** `https://www.youtube.com/watch?v=maY2AobNHXo&t=3322s`
> - **Video:** Cyberpunk 2077 Live-Action Project Announced - Kinda Funny Games Daily 10.05.23
>   - **Timestamp:** `~20m 17s`
>   - **Link:** `https://www.youtube.com/watch?v=WIejox_Iivc&t=1217s`

## Features

- **End-to-End ETL Pipeline:** Manages the entire data lifecycle from raw, disparate sources to a clean, queryable vector database.
- **Retrieval-Augmented Generation (RAG):** Implements a modern RAG architecture for accurate, verifiable question-answering, minimizing LLM hallucinations.
- **LLM-Powered Data Cleaning:** Utilizes local LLMs within the transformation step to intelligently clean and validate unstructured text data.
- **Source-Cited Answers:** The query agent identifies and presents the specific video and timestamp from which information was drawn.
- **Professional Tooling:** Built with a focus on code quality, including 100% strict-mode type checking with Mypy and a comprehensive test suite using Pytest.

## Technology Stack

This project leverages a modern stack of open-source tools to build a robust ETL and AI querying pipeline.

### Core AI & Vector Search

- **LLM Orchestration:** [**LangChain**](https://www.langchain.com/) is used as the central framework to chain together prompts, models, and retrieval systems.
- **Local LLM Inference:** [**Ollama**](https://ollama.com/) serves the large language models locally for all text cleaning, analysis, and question-answering tasks.
- **Embeddings:** [**Hugging Face `sentence-transformers`**](https://huggingface.co/sentence-transformers) are used to generate the high-quality vector embeddings necessary for semantic search.
- **Vector Database:** [**PostgreSQL**](https://www.postgresql.org/) with the [**`pgvector`**](https://github.com/pgvector/pgvector) extension acts as the primary data store, holding both the processed metadata and the vector embeddings.
- **Deep Learning Backend:** [**PyTorch**](https://pytorch.org/) provides the foundational tensor computation for the Hugging Face embedding models.

### Code Quality & Testing

- **Dependency Management:** [**Poetry**](https://python-poetry.org/) for deterministic, reproducible environments.
- **Linting & Formatting:** [**Ruff**](https://github.com/astral-sh/ruff) for high-performance code quality and style enforcement.
- **Testing Framework:** [**Pytest**](https://pytest.org/) with **100% test coverage** for the core application logic.
- **Static Type Checking:** [**Mypy**](http://mypy-lang.org/) configured in `--strict` mode for maximum type safety.
- **Continuous Integration:** [**GitHub Actions**](https://github.com/features/actions) to automatically run all quality checks on every commit.

## Setup & Contributing

This project is maintained with a strict suite of automated quality checks. All contributions are expected to pass these checks.

### Local Development Setup

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/Jer-Pha/kf-ai.git
    cd kf-ai
    ```

2.  **Install Poetry:**
    Follow the [official installation guide](https://python-poetry.org/docs/#installation) to install Poetry on your system.

3.  **Install dependencies:**
    This command will create a virtual environment and install all necessary main and development packages.

    ```bash
    poetry install --all-groups
    ```

4.  **Set up environment variables:**
    Copy the example environment file and fill in your own credentials.

    ```bash
    cp .env.example .env
    ```

5.  **Activate pre-commit hooks:**
    This one-time command installs the Git hooks that will automatically format and lint your code before each commit.
    ```bash
    poetry run pre-commit install
    ```

### Running Quality Checks Manually

You can run all the quality checks at any time from the command line:

```bash
# Run the linter and formatter
poetry run ruff check .
poetry run ruff format .

# Run the static type checker
poetry run mypy --strict

# Run the dependency checker
poetry run deptry .

# Run the full test suite with coverage
poetry run pytest --cov=kfai
```

## Usage

The project is managed via a simple command-line interface. It's designed to be run as an installed package, which registers a clean `kfai` command as the main entry point.

```bash
poetry run kfai
```

This launches the interactive menu for accessing the various pipeline stages:

```text
--- Welcome to KFAI ---

What would you like to do?
  1. Load raw data from KFDB
  2. Process failed video IDs
  3. Transcribe failed videos
  4. Clean raw transcripts
  5. Update the vector store
  6. Interact with the Query Agent (CLI)
  7. Interact with the Query Agent (GUI)
```

## Note on Data Sources

Please be aware that the data extraction step (`extractors/`) is configured to connect to a private database. Therefore, the full pipeline is not runnable out-of-the-box. The primary purpose of this public repository is to serve as a portfolio piece demonstrating modern Python application structure, testing, and AI pipeline implementation.
