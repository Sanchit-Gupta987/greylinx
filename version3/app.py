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
        "pu_0010b1_om_p", "secpu_000bb2_om_p"
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
1. The most relevant tables
2. Relevant parameter names (e.g., SAT, OAT, CHW)
3. If the table is key-value (with `param_id` and `param_value`), suggest how to filter.

Your output should be a **focused SQL task instruction** for the next step, which includes what information we are looking for"

--- HVAC CONTEXT ---
{general_context}

--- USER QUESTION ---
{user_question}

Now write ONLY a focused SQL instruction that provides context by giving the right table(s), fields and if needed param_id vale and an explanation on param_id and param_value. Note: DO NOT GIVE THE SQL ITSELF OR ANY EXTRA EXPLANATION TEXT!
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

Your task is to write 1 or many simple, syntactically correct {dialect} SQL queries that gives all information 
required to help answer the user's question using the given database schema.

Database Schema:
{table_info}

User Questions:
{input}

Generate 1 or many SQL queries below (no explanation):
"""
sql_generation_prompt = PromptTemplate(
    input_variables=["input", "top_k", "table_info", "dialect","user_question"],
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
You are a helpful assistant. Use the SQL query results below and the original user question to generate a clear and concise natural language answer.

Original User Question: {user_question}

SQL Queries and Their Results:
{sql_queries_and_results}

Answer:
"""
answer_prompt = PromptTemplate(
    input_variables=["user_question", "sql_queries_and_results"],
    template=ANSWER_PROMPT_TEMPLATE_STR,
)
try:
    answer_generation_chain = LLMChain(llm=llm, prompt=answer_prompt, verbose=True)
    st.sidebar.success("Answer generation LLMChain ready.")
except Exception as e:
    st.sidebar.error(f"Answer LLMChain init failed: {e}")
    st.stop()

# --- SQL Extraction (Multi-query version) ---
def extract_sql_queries(llm_output: str) -> list[str]:
    queries = []

    # 1. ```sql ...``` blocks
    sql_blocks = re.findall(r"```sql\s*(.*?)\s*```", llm_output, re.DOTALL | re.IGNORECASE)
    for block in sql_blocks:
        split_queries = [q.strip() + ';' for q in block.strip().split(';') if q.strip()]
        queries.extend(split_queries)

    if queries:
        return queries

    # 2. Generic ```...``` blocks
    generic_blocks = re.findall(r"```(.*?)```", llm_output, re.DOTALL)
    for block in generic_blocks:
        if _looks_like_sql(block):
            split_queries = [q.strip() + ';' for q in block.strip().split(';') if q.strip()]
            queries.extend(split_queries)

    if queries:
        return queries

    # 3. Raw fallback parsing
    sql_keywords = ['SELECT', 'WITH']
    statements = re.split(r';\s*', llm_output)
    for stmt in statements:
        stmt_clean = stmt.strip()
        if any(stmt_clean.upper().startswith(k) for k in sql_keywords):
            queries.append(stmt_clean + ';')

    return queries

def _looks_like_sql(text: str) -> bool:
    sql_keywords = ["SELECT", "INSERT", "UPDATE", "DELETE", "WITH", "CREATE", "ALTER", "DROP"]
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
            top_k = 10

            with st.spinner("AI is generating SQL from refined task..."):
                llm_response_dict = sql_generation_chain.invoke({
                    "input": refined_task,
                    "user_question": user_question,
                    "top_k": str(top_k),
                    "table_info": table_info,
                    "dialect": dialect
                })
                raw_llm_output = llm_response_dict.get("text", "")
                st.subheader("LLM Raw Output (SQL Generation):")
                st.text_area("LLM Raw Output:", raw_llm_output, height=150)

                cleaned_sql_queries = extract_sql_queries(raw_llm_output)
                
                if cleaned_sql_queries:
                    all_results_combined = []
                    for i, query in enumerate(cleaned_sql_queries, start=1):
                        st.subheader(f"SQL Query #{i}:")
                        st.code(query, language="sql")

                        try:
                            db_result = db.run(query)
                            result_str = str(db_result)
                            st.markdown(result_str)
                            all_results_combined.append(f"SQL Query #{i}:\n{query}\n\nResult:\n{result_str}\n")
                        except Exception as db_exc:
                            st.error(f"Error executing SQL query #{i}: {db_exc}")
                            all_results_combined.append(f"SQL Query #{i}:\n{query}\n\nResult: Error - {db_exc}\n")

                    st.subheader("Final Answer (Natural Language):")
                    combined_context = "\n\n".join(all_results_combined)
                    answer_response = answer_generation_chain.invoke({
                        "user_question": user_question,
                        "sql_queries_and_results": combined_context
                    })
                    final_answer = answer_response.get("text", "").strip()
                    st.write(final_answer)
                else:
                    st.error("Failed to extract any valid SQL queries from the LLM's response.")
        except Exception as e:
            st.error(f"An error occurred during the process: {e}")
            import traceback
            st.text(traceback.format_exc())

st.sidebar.markdown("---")
st.sidebar.header("⚠️ Important Notes")
st.sidebar.markdown(f"""
- **LLM Model:** This app currently uses `llama3`.
- **Multi-query Support:** LLM responses can include multiple SQL queries (in blocks or plain).
- **`include_tables`:** Limited to relevant `_om_p` HVAC-related tables.
- **Verbose Logs:** Chains use `verbose=True` for terminal-level traceability.
""")
