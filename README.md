# Graylinx - AI-Powered Data Analysis and Visualization

This project was developed during my time at Graylinx to use Large Language Models (LLMs) to help generate reports based on data from a database. The project consists of two main, powerful components: a sophisticated question-answering application and an advanced graphing & analysis pipeline.

## Project Overview

This project provides two distinct Streamlit applications that leverage a locally hosted LLM (Llama 3) to interact with a MySQL database in different ways.

### Component 1: Natural Language Question Answering (`app.py`)

This application is engineered to answer complex, high-level questions by orchestrating a robust, multi-step AI pipeline:

1. **Decomposition:** It first breaks down a high-level question into several smaller, answerable sub-questions.
2. **Refinement:** Each sub-question is then refined, simplifying the general language into a focused instruction to guide the SQL generator.
3. **SQL Generation & Execution:** It generates and runs a SQL query for each refined sub-question.
4. **Sub-Answer Formulation:** It forms a natural language answer for each sub-query's result.
5. **Synthesis:** It synthesizes all the individual sub-answers into a final, coherent, and comprehensive response to the original question.
6. **Report Generation:** It formats this final answer into downloadable Markdown (`.md`) and PDF (`.pdf`) documents.

### Component 2: Graphing & Analysis Pipeline (`graph.py`)

This is an intelligent reporting tool that automatically generates interactive data visualizations and textual analysis from complex natural language questions.
1. **Task Decomposition:** An AI chain intelligently breaks down user requests into specific, atomic sub-tasks suitable for visualization.
2. **Time Frame Resolution:** The decomposer automatically resolves relative time frames (e.g., "last week") into specific, absolute dates.
3. **Dynamic Visualization:** The application generates interactive Plotly graphs, forcing SVG rendering to ensure browser compatibility.
4. **AI-Powered Analysis:** For time-series data, a dedicated "Analyst" AI chain examines the plotted data and generates a concise paragraph interpreting the trends.
5. **Interactive Reporting:** The UI allows users to generate graphs for each task individually and then compile all results into a single, downloadable HTML report.

## Current Setup

- **Database:** The applications connect to a local MySQL database (`sample_db`) using SQLAlchemy.
- **LLM:** The LLaMA 3 model runs locally via the Ollama framework, with LangChain orchestrating the prompts and chains.
- **UI:** User interfaces are built with Streamlit and run in the browser.

## Requirements

- Python 3.10+
- MySQL (running locally or accessible on the network)
- Ollama (with the `llama3` model pulled)
- Python packages: `streamlit`, `langchain-community`, `langchain-ollama`, `sqlalchemy`, `pymysql`, `pandas`, `plotly`, `python-dotenv`, `markdown`, `fpdf` (and any others for PDF generation).

## Running the Apps

0.  Ensure your MySQL server is running and the `sample_db` database is available.

1.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

2.  (If not already running) Start the LLaMA 3 model via Ollama:
    ```bash
    ollama run llama3
    ```

3.  Create a `.env` file in the project root with your database credentials:
    ```
    # The app uses a direct URI string, so these are for reference
    DB_HOST="localhost"
    DB_PORT="3306"
    DB_USERNAME="your_mysql_username"
    DB_PASSWORD="your_mysql_password"
    DB_NAME="sample_db"
    ```
    *Note: Ensure the `db_url` string in both `app.py` and `graph.py` matches your setup, or edit it to use .env variables.*

4.1. Run the Question Answering App:
    ```bash
    streamlit run app.py
    ```

4.2. Run the Graphing & Analysis App:
    ```bash
    streamlit run graph.py
    ```

## Current Features & Notes

- **Question Answering Pipeline (`app.py`):** Implements the full decompose-refine-synthesize strategy to answer complex questions that require multiple database lookups.
- **Multi-Format Export (`app.py`):** Capable of generating and exporting final answers as both Markdown and PDF documents.
- **Graphing Pipeline (`graph.py`):** Uses a multi-chain process for Decomposition, SQL Generation, and Data Analysis to ensure modularity and accuracy.
- **Few-Shot Prompting (`graph.py`):** The Decomposer prompt uses multiple, detailed examples to handle complex queries involving single days, date ranges, and multiple entities.
- **Stateful UI (`graph.py`):** The graphing app uses Streamlit's `session_state` to manage an interactive workflow, allowing for one-by-one generation of items before final report compilation.
- **Conditional AI Analysis (`graph.py`):** AI-generated text analysis is currently implemented and optimized for **Line charts**.

## Next Steps

-   **Enable Abstract Querying:**
    -   Train or fine-tune the model to understand vague, high-level concepts like "performance" (e.g., "How was Building A's performance last month?"). This would involve teaching the model to associate abstract terms with a key set of metrics.

-   **Add Support for Multiple Graph Types (`graph.py`):**
    -   Implement the "Manual Override" UI to allow users to select Bar, Histogram, and Scatter plot options for any sub-task.
    -   Create and integrate specialized SQL generation prompts for each new chart type.

-   **Standardized Report Templating:**
    -   Add functionality to generate a full report in a pre-determined format (e.g., a company-branded PDF or Word template) including titles, summaries, and multiple charts.

-   **Expand AI Analysis (`graph.py`):**
    -   Develop new, specialized analysis prompts tailored to Bar charts, Histograms, and Scatter plots.

-   **Implement Advanced RAG (`graph.py`):**
    -   To move beyond general analysis, implement a Retrieval-Augmented Generation (RAG) pipeline. This would involve building a vector knowledge base from HVAC manuals and operational documents to allow the AI to provide deeply insightful, context-aware analysis.

-   **Improve UI/UX (`graph.py`):**
    -   Enhance error recovery by allowing a user to edit a failed SQL query directly in the UI and re-run it.
    -   Add a "Clear Report" button to manually clear the collected report items from the session state.
