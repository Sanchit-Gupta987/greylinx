import streamlit as st
import os
from dotenv import load_dotenv
import sqlalchemy
import re
from typing import Optional
from langchain_ollama import OllamaLLM as Ollama
from langchain_community.utilities import SQLDatabase
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate

# --- 1. Load Env and Set Config ---
load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2", "true")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "")

# --- 2. Database Connection ---
DB_URI = "mysql+pymysql://root:@localhost:3306/sample_db"
try:
    engine = sqlalchemy.create_engine(DB_URI)
    relevant_tables = [
        "ahu_005ca0_om_p", "ahu_005ca0_om_p__chw_table", "ahu_005ca0_om_p__fan_power_table", "ahu_005ca0_om_p__rpm_table",
        "ahu_005da0_om_p", "ahu_005da0_om_p__chw_table", "ahu_005da0_om_p__fan_power_table", "ahu_005da0_om_p__rpm_table",
        "ahu_005ea0_om_p", "ahu_005ea0_om_p__chw_table", "ahu_005ea0_om_p__fan_power_table", "ahu_005ea0_om_p__rpm_table",
        "ahu_005fa0_om_p", "ahu_005fa0_om_p__chw_table", "ahu_005fa0_om_p__fan_power_table", "ahu_005fa0_om_p__rpm_table",
        "ahu_0060a0_om_p", "ahu_0060a0_om_p__chw_table", "ahu_0060a0_om_p__fan_power_table", "ahu_0060a0_om_p__rpm_table",
        "ahu_0101a0_om_p", "ahu_0101a0_om_p__chw_table", "ahu_0101a0_om_p__fan_power_table", "ahu_0101a0_om_p__rpm_table",
        "ch_000bb0_om_p", "ch_010001b00000_om_p", "ch_010001b00000_om_p__amps_table", "ch_010001b00000_om_p__dis_pre_table",
        "ch_010001b00000_om_p__dis_tem_table", "ch_010001b00000_om_p__suc_pre_table",
        "pu_0010b1_om_p", "reference_om_p", "secpu_000bb2_om_p"
    ]
    db = SQLDatabase(engine, include_tables=relevant_tables)
    st.sidebar.success("Connected to MySQL database!")
    st.sidebar.info(f"Tables considered by LLM: {', '.join(relevant_tables)}")
except Exception as e:
    st.sidebar.error(f"DB connection failed: {e}")
    st.stop()

# --- 3. LLM Setup ---
try:
    llm = Ollama(model="llama3")
    st.sidebar.info("Ollama LLM (llama3) initialized.")
except Exception as e:
    st.sidebar.error(f"LLM init failed: {e}")
    st.stop()

# --- 4. Load General Context from File ---
try:
    with open("basic_full_context.txt", "r") as f:
        general_context = f.read()
except Exception as e:
    st.sidebar.error(f"Failed to load general context: {e}")
    st.stop()

# --- 4a. Refinement Prompt Chain ---
REFINEMENT_PROMPT = """
You are an expert in HVAC systems and SQL databases.

Given the following HVAC database documentation and a user question, identify:
1. The most relevant table(s)
2. Any parameter names (e.g., SAT, OAT, CHW)
3. If the table is key-value (with `param_id` and `param_value`), suggest how to filter.

Your output should be a **focused SQL task instruction** for the next step.

--- HVAC CONTEXT ---
{general_context}

--- USER QUESTION ---
{user_question}

Now write a focused SQL task instruction that maps the question to the right table(s) and fields.
"""
refinement_prompt = PromptTemplate(
    input_variables=["general_context", "user_question"],
    template=REFINEMENT_PROMPT,
)
try:
    refinement_chain = LLMChain(llm=llm, prompt=refinement_prompt, verbose=True)
    st.sidebar.success("Refinement chain (Stage 1) ready.")
except Exception as e:
    st.sidebar.error(f"Refinement chain failed: {e}")
    st.stop()

# --- 5. SQL Generation Prompt ---
CUSTOM_SQL_PROMPT_TEMPLATE_STR = """You are a highly skilled {dialect} SQL expert.

Your task is to write a syntactically correct {dialect} SQL query that answers the user's question using the given database schema.

Return an SQL query wrapped it in ```sql ... ``` with no explanation or extra text.

Database Schema:
{table_info}

User Question:
{input}

Generate a SQL query below (no explanation):
"""
sql_generation_prompt = PromptTemplate(
    input_variables=["input", "top_k", "table_info", "dialect"],
    template=CUSTOM_SQL_PROMPT_TEMPLATE_STR,
)
try:
    sql_generation_chain = LLMChain(llm=llm, prompt=sql_generation_prompt, verbose=True)
    st.sidebar.success("LangChain LLMChain for SQL generation ready.")
except Exception as e:
    st.sidebar.error(f"LLMChain init failed: {e}")
    st.stop()

# --- 6. Answer Generation Chain ---
ANSWER_PROMPT_TEMPLATE_STR = """
You are a helpful assistant. Use the SQL query result below and the original user question to generate a clear and concise natural language answer.

Original User Question: {user_question}

SQL Query Used:
{sql_query}

SQL Query Result:
{query_result}

Answer:"""
answer_prompt = PromptTemplate(
    input_variables=["user_question", "sql_query", "query_result"],
    template=ANSWER_PROMPT_TEMPLATE_STR,
)
try:
    answer_generation_chain = LLMChain(llm=llm, prompt=answer_prompt, verbose=True)
    st.sidebar.success("Answer generation LLMChain ready.")
except Exception as e:
    st.sidebar.error(f"Answer LLMChain init failed: {e}")
    st.stop()

# --- 7. SQL Extraction Helper ---
# THIS SECTION HAS BEEN REPLACED WITH A MORE ROBUST FUNCTION
def extract_sql_query(llm_output: str) -> Optional[str]:
    """
    Extracts a single, multi-line SQL query from the LLM's output.
    Prioritizes ```sql blocks, but falls back to finding the first full SQL statement.
    """
    # Priority 1: Look for a ```sql code block
    match = re.search(r"```sql\s*(.*?)\s*```", llm_output, re.DOTALL | re.IGNORECASE)
    if match:
        query = match.group(1).strip()
        return query if query else None

    # Priority 2: Look for any generic ``` code block
    match = re.search(r"```(.*?)```", llm_output, re.DOTALL)
    if match:
        potential_sql = match.group(1).strip()
        if potential_sql.upper().startswith(('SELECT', 'WITH')):
            return potential_sql

    # Priority 3: If no code blocks, find the first full SQL statement in the text
    # This is more robust than the previous line-by-line approach
    sql_keywords = ['SELECT', 'WITH']
    text_upper = llm_output.upper()
    start_index = -1

    for keyword in sql_keywords:
        found_index = text_upper.find(keyword)
        if found_index != -1:
            if start_index == -1 or found_index < start_index:
                start_index = found_index
    
    if start_index != -1:
        # We found a keyword, now grab everything from there to the end
        full_statement = llm_output[start_index:].strip()
        # Clean up potential trailing text after a semicolon
        if ';' in full_statement:
            full_statement = full_statement.split(';')[0] + ';'
        return full_statement

    st.warning("Could not confidently extract SQL from the LLM's raw output.")
    return None

def _looks_like_sql(text: str) -> bool:
    # This helper function is still useful for the generic code block check
    sql_keywords = ["SELECT", "INSERT", "UPDATE", "DELETE", "WITH", "CREATE", "ALTER", "DROP"]
    # Check if the text starts with any of the keywords
    return any(text.strip().upper().startswith(kw) for kw in sql_keywords)


# --- 8. Streamlit UI ---
st.title("HVAC Data Assistant ⚡")
st.subheader("Ask questions about your HVAC data")

with st.form("query_form"):
    user_question = st.text_area("Your question:", placeholder="e.g., What is the highest recorded temperature.")
    submitted = st.form_submit_button("Ask AI")

if submitted and user_question:
    with st.spinner("AI is refining your question..."):
        try:
            refinement_output = refinement_chain.invoke({
                "general_context": general_context,
                "user_question": user_question
            })
            refined_task = refinement_output.get("text", "").strip()
            st.subheader("Step 1: Refined SQL Task")
            st.text_area("Refined Prompt:", refined_task, height=120)

            table_info = db.get_table_info()
            dialect = db.dialect
            top_k = 10 # This is unused in the prompt, can be removed if desired

            with st.spinner("AI is generating SQL from refined task..."):
                llm_response_dict = sql_generation_chain.invoke({
                    "input": refined_task,
                    "top_k": str(top_k),
                    "table_info": table_info,
                    "dialect": dialect
                })
                raw_llm_output = llm_response_dict.get("text", "")

                st.subheader("LLM Raw Output (SQL Generation):")
                st.text_area("LLM Raw Output:", raw_llm_output, height=150)

                # This should now work correctly with the new function
                cleaned_sql_query = extract_sql_query(raw_llm_output)
                
                if cleaned_sql_query:
                    st.subheader("Extracted SQL Query:")
                    st.code(cleaned_sql_query, language="sql")
                    st.subheader("Database Query Result:")
                    try:
                        db_result = db.run(cleaned_sql_query)
                        result_str = str(db_result)
                        st.markdown(result_str)

                        st.subheader("Final Answer (Natural Language):")
                        answer_response = answer_generation_chain.invoke({
                            "user_question": user_question,
                            "sql_query": cleaned_sql_query,
                            "query_result": result_str
                        })
                        final_answer = answer_response.get("text", "").strip()
                        st.write(final_answer)
                    except Exception as db_exc:
                        st.error(f"Error executing SQL query: {db_exc}")
                        st.warning("The extracted SQL might still be invalid or cause a database error.")
                else:
                    st.error("Failed to extract a valid SQL query from the LLM's response. Please check the raw output above.")
        except Exception as e:
            st.error(f"An error occurred during the process: {e}")
            import traceback
            st.text(traceback.format_exc())

st.sidebar.markdown("---")
st.sidebar.header("⚠️ Important Notes")
st.sidebar.markdown(f"""
- **LLM Model:** This app currently uses `llama3`.
- **`include_tables`:** This app limits LLM access to relevant `_om_p` HVAC-related tables.
- **`verbose=True`:** The `LLMChain` is set to `verbose=True`. Check the terminal for intermediate steps.
- **Two-Stage Prompting:** Step 1 (refine) + Step 2 (SQL generation) improves accuracy.
""")