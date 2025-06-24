import streamlit as st
import os
from dotenv import load_dotenv
import sqlalchemy
import re
import json
from typing import List
from langchain_core.runnables import RunnableSequence
from langchain_ollama import OllamaLLM as Ollama
from langchain_community.utilities import SQLDatabase
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate

import datetime
from io import BytesIO
from fpdf import FPDF

# --- 1. Load Env and Set Config ---
load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2", "true")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "")

# --- 2. Database & LLM Setup (with caching to prevent re-initialization) ---
@st.cache_resource
def initialize_connections():
    """Initializes DB and LLM connections once and caches them."""
    try:
        db_uri = "mysql+pymysql://root:@localhost:3306/sample_db"
        engine = sqlalchemy.create_engine(db_uri)
        relevant_tables = [
            "ahu_005ca0_om_p", "ahu_005ca0_om_p__chw_table", "ahu_005ca0_om_p__fan_power_table", "ahu_005ca0_om_p__rpm_table",
            "ahu_005da0_om_p", "ahu_005da0_om_p__chw_table", "ahu_005da0_om_p__fan_power_table", "ahu_005da0_om_p__rpm_table",
            "ahu_005ea0_om_p", "ahu_005ea0_om_p__chw_table", "ahu_005ea0_om_p__fan_power_table", "ahu_005ea0_om_p__rpm_table",
            "ahu_005fa0_om_p", "ahu_005fa0_om_p__chw_table", "ahu_005fa0_om_p__fan_power_table", "ahu_005fa0_om_p__rpm_table",
            "ahu_0060a0_om_p", "ahu_0060a0_om_p__chw_table", "ahu_0060a0_om_p__fan_power_table", "ahu_0060a0_om_p__rpm_table",
            "ahu_0101a0_om_p", "ahu_0101a0_om_p__chw_table", "ahu_0101a0_om_p__fan_power_table", "ahu_0101a0_om_p__rpm_table",
            "ch_000bb0_om_p", "ch_010001b00000_om_p", "ch_010001b00000_om_p__amps_table", "ch_010001b00000_om_p__dis_pre_table",
            "ch_010001b00000_om_p__dis_tem_table", "ch_010001b00000_om_p__suc_pre_table",
            "pu_0010b1_om_p", "secpu_000bb2_om_p"
        ]
        db = SQLDatabase(engine, include_tables=relevant_tables)
        st.sidebar.success("Connected to MySQL database!")
        st.sidebar.info(f"Monitoring {len(relevant_tables)} tables.")
    except Exception as e:
        st.sidebar.error(f"DB connection failed: {e}")
        return None, None

    try:
        llm = Ollama(model="llama3", temperature=0)
        st.sidebar.info("Ollama LLM (llama3) initialized.")
    except Exception as e:
        st.sidebar.error(f"LLM init failed: {e}")
        return db, None

    return db, llm

db, llm = initialize_connections()

if not db or not llm:
    st.error("Application cannot start due to connection failures.")
    st.stop()

# --- 3. Load General Context from File ---
try:
    with open("basic_full_context.txt", "r") as f:
        general_context = f.read()
except Exception as e:
    st.sidebar.error(f"Failed to load general context: {e}")
    st.stop()

# --- 4. Define All LLM Chains ---

# --- 4a. NEW: Stage 0 - Decomposer Chain ---
# Replace your DECOMPOSER_PROMPT_TEMPLATE with this even stricter version
DECOMPOSER_PROMPT_TEMPLATE = """
You are a meticulous 'Question Parser' AI. Your sole purpose is to deconstruct a user's question into its fundamental, quantitative components. You must act as a parser, not an analyst.

The generated questions MUST follow these strict rules in order of importance:
1.  **COMPLETE AND SELF-CONTAINED:** This is the most important rule. Every single question you generate MUST explicitly restate the metric it is measuring. Do not create ambiguous questions.
2.  **STRICT ADHERENCE TO METRICS:** Decompose the user's question ONLY into the specific metrics and entities explicitly mentioned. DO NOT invent or infer new metrics.
3.  **QUANTITATIVE ONLY:** Only generate questions that can be answered with a number, list, or boolean value. You MUST IGNORE any parts of the user's question that are subjective, qualitative, or ask "why". For example, "What does this tell us about ...?", you should generate NO sub-questions.
4.  **ATOMIC:** Each question must ask for a single metric for a single entity (e.g., "What was the *average temperature for AHU X*?").

Your output **MUST** be a valid JSON array of strings. Do not include any other text, explanation, or markdown.

---
**Example:**

User Question: What was the average Return Air Temperature and the max Fan Power for AHU 5da0 and AHU 5ea0 last week?

JSON Array of Sub-questions:
[
    "What was the average Return Air Temperature for ahu_005da0_om_p last week?",
    "What was the average Return Air Temperature for ahu_005ea0_om_p last week?",
    "What was the maximum Fan Power for ahu_005da0_om_p last week?",
    "What was the maximum Fan Power for ahu_005ea0_om_p last week?"
]
---

**Task:**

User Question: {complex_question}

JSON Array of Sub-questions:
"""

decomposer_prompt = PromptTemplate(input_variables=["complex_question"], template=DECOMPOSER_PROMPT_TEMPLATE)
decomposer_chain = decomposer_prompt | llm

# --- 4b. The Original Pipeline X Chains (IMPROVED PROMPTS) ---

REFINEMENT_PROMPT = """
You are an HVAC + SQL expert. Your task is to convert a user's question into a precise instruction for generating a SQL query.

**Your Output Must:**
- Identify only the exact table(s) and field(s) required.
- Specify the filtering logic using correct column names. For key-value tables, include valid `param_id` values from the documentation.
- Clearly define the expected output (e.g., average, count, latest value). If raw values are needed, apply a LIMIT of 100 unless told otherwise.
- Avoid unnecessary elaboration, summaries, or SQL code. Keep it short and directive.

OUTPUT FORMAT:
Instruction : <Concise definition of what is needed>
Table(s) : <Relevant tables>
Column names : <Relevant Columns>
Filter conditions : <Filter conditions including any Param_id-Param_value key-value pairs>
Final instruction on output : <Exactly what's needed - Average / Max / Count / Select *>

--- DATABASE DOCUMENTATION ---
{general_context}

--- USER QUESTION ---
{user_question}

--- SQL TASK INSTRUCTION ---
"""

refinement_prompt = PromptTemplate(input_variables=["general_context", "user_question"], template=REFINEMENT_PROMPT)
refinement_chain = refinement_prompt | llm

# Replace your old CUSTOM_SQL_PROMPT_TEMPLATE_STR with this one
CUSTOM_SQL_PROMPT_TEMPLATE_STR = """
You are a specialized AI that translates a structured task description into a single, executable MySQL query.

**Your Task:**
Analyze the provided DATABASE SCHEMA and the STRUCTURED TASK below. Your SOLE output must be a single, complete, and syntactically correct MySQL query that accomplishes the task.

**CRITICAL RULES:**
1.  **SQL ONLY:** Your output MUST be ONLY the raw MySQL query. Do not include explanations, comments, or markdown formatting like ```sql.
2.  **NO CONVERSATION:** Do not write any greetings or text that is not part of the executable SQL query.
3.  **FOLLOW THE TASK:** Mechanically translate the components from the STRUCTURED TASK into a query.
4.  **MANDATORY LIMIT FILTER:** If the task asks for an entire table, set a limit of the 100 latest entries!
5.  **SINGLE QUERY ONLY:** Do not generate multiple alternative queries or provide variations.

--- DATABASE SCHEMA ---
{table_info}

--- STRUCTURED TASK ---
{input}

--- MySQL QUERY ---
"""

sql_generation_prompt = PromptTemplate(input_variables=["input", "table_info", "dialect"], template=CUSTOM_SQL_PROMPT_TEMPLATE_STR)
sql_generation_chain = sql_generation_prompt | llm


ANSWER_PROMPT_TEMPLATE_STR = """
You are a helpful assistant. Use the SQL query results below to generate a clear, concise and specific natural language answer for the specific question provided.

I want no other analysis on the SQL queries and their outputs. Simply pick the information that is valid and use it to ONLY answer the question.

Specific Question to Answer: {user_question}

SQL Queries and Their Results:
{sql_queries_and_results}

Answer:
"""
answer_prompt = PromptTemplate(input_variables=["user_question", "sql_queries_and_results"], template=ANSWER_PROMPT_TEMPLATE_STR)
answer_generation_chain = answer_prompt | llm

# --- 4c. NEW: Final Stage - Aggregator Chain ---
AGGREGATOR_PROMPT_TEMPLATE = """
You are a helpful assistant. You have been given a user's original question and a series of answers to simpler sub-questions. Combine these individual answers into a single, cohesive response.

Original User Question: {original_question}

Individual Answers Found:
---
{individual_answers}
---
Combine the findings into a final, complete answer:
"""
aggregator_prompt = PromptTemplate(input_variables=["original_question", "individual_answers"], template=AGGREGATOR_PROMPT_TEMPLATE)
aggregator_chain = aggregator_prompt | llm


# --- 5. SQL Extraction Helper Function (REPLACE THE OLD ONE WITH THIS) ---

def _looks_like_sql(text: str) -> bool:
    """A helper function to check if a string looks like a SQL query."""
    sql_keywords = ["SELECT", "INSERT", "UPDATE", "DELETE", "WITH", "CREATE", "ALTER", "DROP"]
    # Check for keywords at the beginning of the string, ignoring leading whitespace
    return any(text.strip().upper().startswith(kw) for kw in sql_keywords)

def extract_sql_queries(llm_output: str) -> list[str]:
    """
    Extracts SQL queries from an LLM's output using a robust, multi-stage approach.
    """
    queries = []

    # 1. Primary Method: Find explicit ```sql ...``` blocks
    sql_blocks = re.findall(r"```sql\s*(.*?)\s*```", llm_output, re.DOTALL | re.IGNORECASE)
    for block in sql_blocks:
        # Split by semicolon and filter out empty strings
        split_queries = [q.strip() for q in block.strip().split(';') if q.strip()]
        # Add the semicolon back to each valid query
        queries.extend([f"{q};" for q in split_queries])

    if queries:
        return queries

    # 2. Secondary Method: Find generic ```...``` blocks and check if they contain SQL
    generic_blocks = re.findall(r"```(.*?)```", llm_output, re.DOTALL)
    for block in generic_blocks:
        if _looks_like_sql(block):
            split_queries = [q.strip() for q in block.strip().split(';') if q.strip()]
            queries.extend([f"{q};" for q in split_queries])

    if queries:
        return queries

    # 3. Fallback Method: Parse raw text for statements starting with SQL keywords
    sql_keywords = ['SELECT', 'WITH']
    # Use a positive lookbehind to split on semicolons, keeping the delimiter
    statements = re.split(r'(?<=\;)\s*', llm_output)
    for stmt in statements:
        stmt_clean = stmt.strip()
        if any(stmt_clean.upper().startswith(k) for k in sql_keywords):
            # Ensure it ends with a semicolon
            if not stmt_clean.endswith(';'):
                stmt_clean += ';'
            queries.append(stmt_clean)
            
    # Remove duplicates while preserving order
    if queries:
        return list(dict.fromkeys(queries))

    return []


# --- 6. KEY CHANGE: Refactored Pipeline X into a Reusable Function (WITH UI MODS) ---
def run_single_question_pipeline(question_to_run: str) -> str:
    """
    Runs the complete Refine -> SQL -> Execute -> Answer pipeline for a SINGLE simple question.
    Returns the natural language answer as a string.
    """
    try:
        with st.expander(f"Processing Sub-Question: '{question_to_run}'"):
            # Stage 1: Refine
            st.write("➡️ Stage 1: Refining task...")
            refined_task = refinement_chain.invoke({"general_context": general_context, "user_question": question_to_run}).strip()
            st.text_area("Refined Task:", refined_task, height=150, key=f"refined_{question_to_run}")

            # Stage 2: Generate SQL
            st.write("➡️ Stage 2: Generating SQL...")
            table_info = db.get_table_info()
            dialect = db.dialect
            raw_sql_output = sql_generation_chain.invoke({"input": refined_task, "table_info": table_info, "dialect": dialect}).strip()

            # --- NEW: Show the raw LLM output for transparency ---
            st.text_area("LLM Raw SQL Output:", raw_sql_output, height=150, key=f"raw_sql_{question_to_run}")

            cleaned_sql_queries = extract_sql_queries(raw_sql_output)

            if not cleaned_sql_queries:
                st.error("Failed to generate/extract SQL for this sub-question.")
                return f"Could not answer '{question_to_run}' because SQL generation failed."

            st.write("✅ **Extracted SQL Query/Queries:**")
            st.code('\n---\n'.join(cleaned_sql_queries), language="sql")

            # Stage 3: Execute SQL
            st.write("➡️ Stage 3: Executing SQL and gathering results...")
            all_results_combined = []

            # --- NEW: Cleaner loop for displaying results ---
            for i, query in enumerate(cleaned_sql_queries):
                st.markdown(f"--- \n##### Query {i+1}:")
                st.code(query, language="sql")
                try:
                    db_result = db.run(query)
                    st.text_area("Result:", value=str(db_result), height=100, key=f"result_{question_to_run}_{i}")
                    all_results_combined.append(f"Query:\n{query}\nResult:\n{db_result}")
                except Exception as db_exc:
                    error_message = f"Error executing query: {db_exc}"
                    st.error(error_message)
                    all_results_combined.append(f"Query:\n{query}\nError:\n{error_message}")

            combined_context = "\n\n".join(all_results_combined)

            # Stage 4: Generate Final Answer for this sub-question
            st.write("➡️ Stage 4: Synthesizing answer for sub-question...")
            final_answer_for_sub_q = answer_generation_chain.invoke({"user_question": question_to_run, "sql_queries_and_results": combined_context}).strip()
            st.success(f"Sub-answer: {final_answer_for_sub_q}")
            return final_answer_for_sub_q

    except Exception as e:
        st.error(f"An error occurred while processing '{question_to_run}': {e}")
        return f"An error occurred while trying to answer: '{question_to_run}'."


# --- 7. NEW: Main Streamlit UI and Orchestrator Logic ---
st.title("Advanced HVAC Data Assistant 🧠")
st.subheader("Powered by a Decomposer-Aggregator AI Pipeline")

with st.form("query_form"):
    user_question = st.text_area("Your question:", placeholder="e.g., For AHU 5ca0, what is the average RPM and the highest Fan Power consumption?")
    submitted = st.form_submit_button("🚀 Ask AI")

if submitted and user_question:
    # --- STAGE 0: DECOMPOSE ---
    st.subheader("1. Decomposer Stage")
    with st.spinner("Breaking down your complex question..."):
        decomposer_output = decomposer_chain.invoke({"complex_question": user_question})
        decomposed_text = decomposer_output.strip()

        # Robustly parse the JSON output
        try:
            # Handle cases where the LLM might add ```json ... ```
            if decomposed_text.startswith("```json"):
                decomposed_text = re.search(r"```json\s*(.*?)\s*```", decomposed_text, re.DOTALL).group(1)
            sub_questions = json.loads(decomposed_text)
            if not isinstance(sub_questions, list): # Ensure it's a list
                 raise json.JSONDecodeError("Output is not a list", decomposed_text, 0)
        except (json.JSONDecodeError, AttributeError):
            st.warning("Could not decompose the question. Proceeding with the original question as a single task.")
            sub_questions = [user_question] # Fallback

        st.write("Identified sub-questions:")
        st.write(sub_questions)

    # --- STAGE 1-N: MAP & PROCESS ---
    st.subheader(f"2. Processing {len(sub_questions)} Sub-Question(s)")
    all_sub_answers = []
    for i, sub_q in enumerate(sub_questions):
        # Call the refactored pipeline for each sub-question
        answer_text = run_single_question_pipeline(sub_q)
        all_sub_answers.append(f"Answer to '{sub_q}':\n{answer_text}")

    # --- FINAL STAGE: AGGREGATE ---
    if len(all_sub_answers) > 0:
        st.subheader("3. Aggregator Stage")
        with st.spinner("Combining all findings into a final answer..."):
            formatted_answers = "\n\n---\n\n".join(all_sub_answers)

            final_response = aggregator_chain.invoke({
                "original_question": user_question,
                "individual_answers": formatted_answers
            })
            final_answer = final_response.strip()

            # Display final answer
            st.markdown("---")
            st.header("✅ Final Comprehensive Answer")
            st.markdown(final_answer)

            # --- Markdown download ---
            final_output_md = f"# Final Answer\n\n**Question:** {user_question}\n\n**Answer:**\n\n{final_answer}"
            filename_md = f"hvac_answer_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.md"

            st.download_button(
                label="📥 Download as Markdown (.md)",
                data=final_output_md,
                file_name=filename_md,
                mime="text/markdown"
            )

            # --- PDF download ---
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            pdf.multi_cell(0, 10, f"Final Answer\n\nQuestion: {user_question}\n\nAnswer:\n{final_answer}")

            pdf_bytes = pdf.output(dest='S').encode('latin1')
            pdf_buffer = BytesIO(pdf_bytes)
            pdf_buffer.seek(0)

            filename_pdf = f"hvac_answer_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"

            st.download_button(
                label="📥 Download as PDF (.pdf)",
                data=pdf_buffer,
                file_name=filename_pdf,
                mime="application/pdf"
            )
