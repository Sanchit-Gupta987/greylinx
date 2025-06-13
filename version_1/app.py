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

# --- 4. Custom Prompt Template for SQL Generation ---
CUSTOM_SQL_PROMPT_TEMPLATE_STR = """You are a highly skilled {dialect} SQL expert.

Your task is to write a syntactically correct {dialect} SQL query that answers the user's question using the given database schema.

ONLY return the SQL query. If you use markdown, wrap it in ```sql ... ``` with no explanation or extra text.

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

# --- 5. LLMChain for SQL Generation ---
try:
    sql_generation_chain = LLMChain(llm=llm, prompt=sql_generation_prompt, verbose=True)
    st.sidebar.success("LangChain LLMChain for SQL generation ready.")
except Exception as e:
    st.sidebar.error(f"LLMChain init failed: {e}")
    st.stop()

# --- 5b. Prompt and Chain for Natural Language Answer ---
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

# --- 6. Function to Extract SQL from LLM Output ---
def extract_sql_query(llm_output: str) -> Optional[str]:
    match_sql_block = re.search(r"```sql\s*(.*?)\s*```", llm_output, re.DOTALL | re.IGNORECASE)
    if match_sql_block:
        return match_sql_block.group(1).strip()
    match_generic_block = re.search(r"```\s*(.*?)\s*```", llm_output, re.DOTALL)
    if match_generic_block:
        code_block = match_generic_block.group(1).strip()
        if _looks_like_sql(code_block):
            return code_block
    if "SQLQuery:" in llm_output:
        llm_output = llm_output.split("SQLQuery:", 1)[1].strip()
    lines = llm_output.strip().splitlines()
    sql_keywords = ["SELECT", "INSERT", "UPDATE", "DELETE", "WITH", "CREATE", "ALTER", "DROP"]
    for line in lines:
        stripped = line.strip()
        if any(stripped.upper().startswith(kw) for kw in sql_keywords):
            if ';' in stripped:
                return stripped.split(';')[0].strip() + ';'
            else:
                return stripped
    st.warning("Could not confidently extract SQL. The LLM might have included extra text.")
    return None

def _looks_like_sql(text: str) -> bool:
    sql_keywords = ["SELECT", "INSERT", "UPDATE", "DELETE", "WITH", "CREATE", "ALTER", "DROP"]
    first_line = text.strip().splitlines()[0] if text.strip() else ""
    return any(first_line.upper().startswith(kw) for kw in sql_keywords)

# --- 7. Streamlit UI ---
st.title("HVAC Data Assistant ⚡")
st.subheader("Ask questions about your HVAC data")

with st.form("query_form"):
    user_question = st.text_area("Your question:", placeholder="e.g., What is the highest recorded temperature.")
    submitted = st.form_submit_button("Ask AI")

if submitted and user_question:
    with st.spinner("AI is generating SQL and querying the database..."):
        try:
            table_info = db.get_table_info()
            dialect = db.dialect
            top_k = 10

            llm_response_dict = sql_generation_chain.invoke({
                "input": user_question,
                "top_k": str(top_k),
                "table_info": table_info,
                "dialect": dialect
            })
            raw_llm_output = llm_response_dict.get("text", "")

            st.subheader("LLM Raw Output (Attempting to Generate SQL):")
            st.text_area("LLM Raw Output:", raw_llm_output, height=150)

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
                    try:
                        answer_response = answer_generation_chain.invoke({
                            "user_question": user_question,
                            "sql_query": cleaned_sql_query,
                            "query_result": result_str
                        })
                        final_answer = answer_response.get("text", "").strip()
                        st.write(final_answer)
                    except Exception as answer_exc:
                        st.error(f"Error generating final answer: {answer_exc}")
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
- **Custom Prompt & Parsing:** This version uses a stricter custom prompt and attempts to parse the SQL from the LLM's output before execution.
""")