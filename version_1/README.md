# Version 1: Two-Chain Text-to-SQL System

This version implements a functional text-to-SQL pipeline using two distinct LLM chains for improved reliability.

### How It Works

1.  **SQL Generation:** The first LLM chain converts the user's natural language question into a SQL query.
2.  **SQL Parsing:** A custom function (`extract_sql_query`) cleans the raw LLM output to isolate the executable SQL code.
3.  **Answer Generation:** After running the query, a second LLM chain uses the query results to formulate a final, human-readable answer.

### Key Characteristics

* **Two-Chain Architecture:** Separates the logic for SQL generation from final answer generation.
* **Hardcoded Configuration:** The database connection string and the list of tables the LLM can see are hardcoded directly in the Python script.
* **Transparent UI:** The Streamlit interface displays the raw LLM output, the extracted query, the database results, and the final answer for debugging.

## Notes
* LLM Currently doesn't fully understand the database properly and context window sometimes gets overloaded.
