### `version_2/README.md`

## Version 2: Three-Stage Prompt Chaining & External Context

This version significantly enhances the pipeline by introducing a preliminary "Refinement" stage. This improves SQL generation accuracy by using an external context file to better understand the user's intent before creating the query.

### How It Works (Updated Flow)

The process is now a three-stage chain:

1.  **Context Loading (New):** The application first loads general HVAC documentation from an external file (`basic_full_context.txt`).
2.  **Question Refinement (New Stage 1):** A new "Refinement Chain" uses the external context and the user's raw question to create a detailed, focused task instruction. This maps the user's intent to specific database tables and parameters.
3.  **SQL Generation (Stage 2):** The SQL Generation chain now receives this *refined task* instead of the original question, leading to more accurate and reliable SQL queries.
4.  **Answer Generation (Stage 3):** The final chain uses the query results to create a natural-language answer, as before.

### Key Characteristics & Improvements

* **Three-Stage Pipeline:** The new Refine -> Generate SQL -> Generate Answer workflow improves overall accuracy by breaking the problem down.
* **External Context File:** Domain-specific knowledge is now loaded from `basic_full_context.txt`, making the system more modular and easier to update.
* **Improved SQL Parser:** The `extract_sql_query` function has been upgraded to be more robust in handling various LLM output formats.
* **Enhanced UI:** The Streamlit interface now visualizes all stages of the new process, including the "Refined SQL Task" step for better transparency.