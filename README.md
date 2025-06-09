# Graylinx

This project was developed during my time at Graylinx to use LLMs to help generate report based on data from a database.

## Project Overview

This is a local Streamlit application that connects to a MySQL database and uses a locally hosted LLM (Llama3) to:
- Convert natural language questions into SQL queries
- Execute those queries on a database
- Return results in both tabular and natural language format

## Current Setup

1. Database is stored locally (`sample_db`) and accessed via SQLAlchemy.
2. The LLaMA3 model runs locally through the Ollama framework.
3. The UI is built with Streamlit and runs in the browser.

## Requirements

- Python 3.10+
- MySQL (running locally)
- Ollama (model: `llama3`)
- Python packages listed in `requirements.txt`

## Running the App

0. Have the Database available through mySQL.

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Start the LLaMA3 model:
   ```
   ollama run llama3
   ```

3. Create a `.env` file:
   ```
   LANGCHAIN_API_KEY=
   LANGCHAIN_PROJECT=

   DB_TYPE="mysql"
   DB_HOST="localhost"
   DB_PORT="3306"
   DB_USERNAME=
   DB_PASSWORD=
   DB_NAME="sample_db"
   ```

4. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

## Notes

- The current database connection string assumes MySQL is running locally.
- The application parses raw LLM output to extract SQL. This may fail if the model returns unexpected text.
- Only basic error handling is currently implemented.
- Currently handles a single SQL output.

## Next Steps

- Improve SQL generation by adding sample data to prompts or RAG pipeline?
- Include data previews or statistics in the prompt
- Improve query validation before execution
- Multiple query support
- Generation of report

