import streamlit as st
import os
from dotenv import load_dotenv
import sqlalchemy
import re
import json # --- NEW: Imported for parsing LLM output
from typing import List

# --- Make sure to install langchain if you haven't already
# pip install langchain langchain-community langchain-core langchain-ollama
from langchain_ollama import OllamaLLM as Ollama
from langchain_community.utilities import SQLDatabase
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate

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
        llm = Ollama(model="llama3")
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
DECOMPOSER_PROMPT_TEMPLATE = """
You are an expert **HVAC data analyst** who specializes in breaking down complex user requests into a series of atomic, quantitative questions. Your purpose is to generate questions that can be answered directly and independently by a single query to a time-series database containing HVAC performance data (e.g., temperature, humidity, energy consumption, setpoints, runtime).

The generated questions MUST follow these rules:
1.  **Atomic:** Each question must ask for a single metric and a single aggregation (e.g., "What was the *average temperature*?", NOT "What were the average temperature and humidity?").
2.  **No Comparisons:** Do not ask for comparisons or relationships within a single question. A comparison like "Compare Unit A and Unit B" should be decomposed into separate questions for Unit A's metric and Unit B's metric.
3.  **Quantitative & Factual:** Questions must be answerable with a number, list, or boolean value from a database. Avoid subjective questions or questions that ask "why".
4.  **Self-Contained:** Each question should contain all the necessary context (e.g., location, time frame, equipment ID) mentioned in the user's request.

Your output **MUST** be a valid JSON array of strings. Do not include any other text, explanation, markdown, or ```json``` wrappers.

---
**Example:**

User Question: Is the AHU in the south wing running efficiently last week, and how does it compare to the north wing's unit?

JSON Array of Sub-questions:
[
    "What was the average daily energy consumption in kWh for the south wing AHU last week?",
    "What was the average daily energy consumption in kWh for the north wing AHU last week?",
    "What was the average supply air temperature for the south wing AHU last week?",
    "What was the average return air temperature for the south wing AHU last week?",
    "What was the average supply air temperature for the north wing AHU last week?",
    "What was the average return air temperature for the north wing AHU last week?",
    "What was the total runtime in hours for the south wing AHU last week?",
    "What was the total runtime in hours for the north wing AHU last week?"
]
---

**Task:**

User Question: {complex_question}

JSON Array of Sub-questions:
"""
decomposer_prompt = PromptTemplate(input_variables=["complex_question"], template=DECOMPOSER_PROMPT_TEMPLATE)
decomposer_chain = LLMChain(llm=llm, prompt=decomposer_prompt)

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
Column names : <Columns>
Filter conditions : <Filter conditions including any Param_id-Param_value key-value pairs>
Final instruction on output : <Exactly what's needed - Average / Max / Count / Select *>

--- DATABASE DOCUMENTATION ---
{general_context}

--- USER QUESTION ---
{user_question}

--- SQL TASK INSTRUCTION ---
"""

refinement_prompt = PromptTemplate(input_variables=["general_context", "user_question"], template=REFINEMENT_PROMPT)
refinement_chain = LLMChain(llm=llm, prompt=refinement_prompt)

CUSTOM_SQL_PROMPT_TEMPLATE_STR = """
You are a highly skilled {dialect} SQL expert and HVAC data analyst.

You will write one or more syntactically correct {dialect} SQL queries to extract the requested information from the HVAC database.

You are provided with:
1. The full database schema below — treat it as ground truth. Do NOT use any column, table, or param_id that is not explicitly present here.
2. A refined task from a previous step — use it as guidance, but double-check any assumptions against the schema.

Your goal is to generate SQL queries that fulfill the user’s intent.

IMPORTANT RULES:
- Never return the full contents of a table unless explicitly requested.
- Always include filtering conditions (e.g., on `param_id`, `measured_time`, or equipment IDs).
- If the question does not ask for aggregation (e.g., AVG, MAX, COUNT), apply a `LIMIT 100` clause to avoid large result sets.
- Avoid selecting columns or param_ids not mentioned in the documentation or the refined instruction.

--- SCHEMA ---
{table_info}

--- TASK TO IMPLEMENT ---
{input}

Write one or more SELECT queries (no explanation):
"""

sql_generation_prompt = PromptTemplate(input_variables=["input", "table_info", "dialect"], template=CUSTOM_SQL_PROMPT_TEMPLATE_STR)
sql_generation_chain = LLMChain(llm=llm, prompt=sql_generation_prompt)


ANSWER_PROMPT_TEMPLATE_STR = """
You are a helpful assistant. Use the SQL query results below to generate a clear and concise natural language answer for the specific question provided.

Specific Question to Answer: {user_question}

SQL Queries and Their Results:
{sql_queries_and_results}

Answer:
"""
answer_prompt = PromptTemplate(input_variables=["user_question", "sql_queries_and_results"], template=ANSWER_PROMPT_TEMPLATE_STR)
answer_generation_chain = LLMChain(llm=llm, prompt=answer_prompt)

# --- 4c. NEW: Final Stage - Aggregator Chain ---
# --- NEW: Final Stage - Aggregator Chain ---
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
aggregator_chain = LLMChain(llm=llm, prompt=aggregator_prompt)


# --- 5. SQL Extraction Helper Function ---
def extract_sql_queries(llm_output: str) -> List[str]:
    # This function is the same as before, no changes needed.
    queries = re.findall(r"```sql\s*(.*?)\s*```", llm_output, re.DOTALL | re.IGNORECASE)
    if queries:
        return [q.strip() for q in queries[0].split(';') if q.strip()]
    # Fallback for plain code
    return [q.strip() for q in llm_output.split(';') if q.strip().upper().startswith("SELECT")]

# --- 6. KEY CHANGE: Refactored Pipeline X into a Reusable Function ---
def run_single_question_pipeline(question_to_run: str) -> str:
    """
    Runs the complete Refine -> SQL -> Execute -> Answer pipeline for a SINGLE simple question.
    Returns the natural language answer as a string.
    """
    try:
        with st.expander(f"Processing Sub-Question: '{question_to_run}'"):
            # Stage 1: Refine
            st.write("➡️ Stage 1: Refining task...")
            refinement_output = refinement_chain.invoke({"general_context": general_context, "user_question": question_to_run})
            refined_task = refinement_output.get("text", "").strip()
            st.text_area("Refined Task:", refined_task, height=100, key=f"refined_{question_to_run}")

            # Stage 2: Generate SQL
            st.write("➡️ Stage 2: Generating SQL...")
            table_info = db.get_table_info()
            dialect = db.dialect
            sql_response = sql_generation_chain.invoke({"input": refined_task, "table_info": table_info, "dialect": dialect})
            raw_sql_output = sql_response.get("text", "")
            cleaned_sql_queries = extract_sql_queries(raw_sql_output)
            
            if not cleaned_sql_queries:
                st.error("Failed to generate SQL for this sub-question.")
                return f"Could not answer '{question_to_run}' because SQL generation failed."

            st.code('\n'.join(cleaned_sql_queries), language="sql")

            # Stage 3: Execute SQL
            st.write("➡️ Stage 3: Executing SQL and gathering results...")
            all_results_combined = []
            for query in cleaned_sql_queries:
                try:
                    db_result = db.run(query)
                    all_results_combined.append(f"Query:\n{query}\nResult:\n{db_result}")
                except Exception as db_exc:
                    all_results_combined.append(f"Query:\n{query}\nError:\n{db_exc}")
            
            combined_context = "\n\n".join(all_results_combined)
            st.text_area("SQL Results:", combined_context, height=150, key=f"results_{question_to_run}")

            # Stage 4: Generate Final Answer for this sub-question
            st.write("➡️ Stage 4: Synthesizing answer for sub-question...")
            answer_response = answer_generation_chain.invoke({"user_question": question_to_run, "sql_queries_and_results": combined_context})
            final_answer_for_sub_q = answer_response.get("text", "No answer could be generated.").strip()
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
        decomposed_text = decomposer_output.get("text", "[]").strip()
        
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
            final_answer = final_response.get("text", "Could not generate a final summarized answer.")
            
            st.markdown("---")
            st.header("✅ Final Comprehensive Answer")
            st.markdown(final_answer)
