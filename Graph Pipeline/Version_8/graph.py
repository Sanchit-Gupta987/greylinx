import streamlit as st
import os
from dotenv import load_dotenv
import sqlalchemy
import re
import json
import pandas as pd
import plotly.express as px
from io import BytesIO
import datetime
import textwrap
from langchain_core.prompts import PromptTemplate
from langchain_community.utilities import SQLDatabase
from langchain_ollama import OllamaLLM as Ollama

# --- 1. Load Environment ---
load_dotenv()

# --- 2. Database and LLM Setup ---
@st.cache_resource
def initialize():
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
        llm = Ollama(model="llama3", temperature=0)
        st.sidebar.success("Database and LLM initialized successfully!")
        return db, llm
    except Exception as e:
        st.sidebar.error(f"Initialization failed: {e}")
        return None, None

db, llm = initialize()

# --- 3. Load General Context ---
try:
    with open("basic_full_context.txt", "r") as f:
        general_context = f.read()
except FileNotFoundError:
    general_context = "No context file found."
    st.sidebar.warning(general_context)

# --- 4. Prompt Chains ---
DECOMPOSER_PROMPT_TEMPLATE = """
You are a meticulous 'Question Parser' AI. Your sole purpose is to deconstruct a user's request for a graph into its fundamental, quantitative components.
Each component should be a graphable time-series question.

The generated questions MUST follow these strict rules:
1.  **COMPLETE AND SELF-CONTAINED:** Every single question you generate MUST explicitly restate the metric it is measuring.
2.  **STRICT ADHERENCE TO METRICS:** Decompose the user's question ONLY into the specific metrics and entities explicitly mentioned.
3.  **QUANTITATIVE ONLY:** Only generate questions that can be answered with a time-series graph. IGNORE any subjective parts of the user's question.
4.  **ATOMIC:** Each question must ask for a single metric for a single entity.

Your output **MUST** be a valid JSON array of strings. Do not include any other text, explanation, or markdown.
---
**Example:**
User Question: Show me a trend of the Return Air Temperature and the Fan Power for AHU 5da0 last week.
JSON Array of Sub-questions:
[
    "What was the trend of Return Air Temperature for ahu_005da0_om_p last week?",
    "What was the trend of Fan Power for ahu_005da0_om_p last week?"
]
---
**Task:**
User Question: {complex_question}
JSON Array of Sub-questions:
"""
decomposer_chain = PromptTemplate(input_variables=["complex_question"], template=DECOMPOSER_PROMPT_TEMPLATE) | llm

GRAPH_SQL_PROMPT_TEMPLATE = """
You are an expert data analyst who writes syntactically correct MySQL queries for generating time-series line graphs.
Your SOLE purpose is to write a single MySQL query that retrieves data suitable for a line chart based on the user's question and the provided schema.

**CRITICAL RULES:**
1.  **NEVER Invent Columns:** You must only use columns that explicitly exist in the table schema provided. Do not hallucinate column names like `timestamp` or `RAT`.
2.  **ALWAYS ALIAS COLUMNS:** The final query **MUST** select exactly two columns with specific aliases:
    - The first column must be the time column (e.g., `measured_time`) and **MUST** be aliased as `timestamp`.
    - The second column must be the numeric data column and **MUST** be aliased as `value`.
3.  **KEY-VALUE TABLE LOGIC:** For key-value tables like `ahu_..._om_p`, the metric (e.g., 'Return Air Temperature') corresponds to a `param_id`. To get its value, you **MUST**:
    - Add a `WHERE param_id = '...'` clause.
    - `SELECT` the `param_value` column and alias it `AS value`.
4.  **SORT BY TIME:** The results **MUST** be sorted by the timestamp in ascending order (`ORDER BY timestamp ASC`).
5.  **SQL ONLY:** Your output **MUST** be ONLY the raw MySQL query. No explanations or markdown.

---
**EXAMPLE OF CORRECT QUERY PATTERN:**

USER QUESTION for the graph: What was the trend of Return Air Temperature for ahu_005ca0 on 2024-01-11?

CORRECT MySQL Query:
SELECT measured_time AS timestamp, param_value AS value FROM ahu_005ca0_om_p WHERE param_id = 'RAT' AND DATE(measured_time) = '2024-01-11' ORDER BY timestamp ASC;
---

**DATABASE CONTEXT:**
{general_context}

**USER QUESTION for the graph:**
{graph_question}

--- MySQL Query for a Line Graph ---
"""
sql_generation_chain = PromptTemplate(input_variables=["general_context", "graph_question"], template=GRAPH_SQL_PROMPT_TEMPLATE) | llm

# --- 5. SQL Extraction Helper Function ---
def _looks_like_sql(text: str) -> bool:
    """A helper function to check if a string looks like a SQL query."""
    return any(text.strip().upper().startswith(kw) for kw in ["SELECT", "WITH"])

def extract_sql(llm_output: str) -> str:
    """Extracts a single SQL query from an LLM's output using a robust, multi-stage approach."""
    llm_output = llm_output.strip()
    match = re.search(r"```sql\s*(.*?)\s*```", llm_output, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    match = re.search(r"```(.*?)```", llm_output, re.DOTALL)
    if match:
        potential_sql = match.group(1).strip()
        if _looks_like_sql(potential_sql):
            return potential_sql
    if _looks_like_sql(llm_output):
        return llm_output
    return llm_output

# --- 6. Streamlit UI ---
st.title("📊 HVAC Graphing Assistant")
st.markdown("Ask for a time-series trend and the AI will generate the corresponding graphs.")

question = st.text_area("Ask a graphable question:",
                        placeholder="Trend of return air temp and supply air temp for AHU 005ca0 for Jan 11, 2024")

if st.button("Generate Graphs") and question:
    if not db or not llm:
        st.error("Database or LLM not initialized. Please check your connection.")
        st.stop()

    st.subheader("1. Decomposing Question")
    with st.spinner("Breaking down question..."):
        decomposed_raw = decomposer_chain.invoke({"complex_question": question})
        try:
            if "```json" in decomposed_raw:
                decomposed_raw = re.search(r"```json\s*(.*?)\s*```", decomposed_raw, re.DOTALL).group(1)
            sub_questions = json.loads(decomposed_raw)
        except (json.JSONDecodeError, AttributeError, TypeError):
            st.warning("Could not decompose the question. Using the original question as a single task.")
            sub_questions = [question]

    st.write("Identified sub-tasks for graphing:")
    st.json(sub_questions)
    
    figures_to_export = []

    st.subheader("2. Generating Graphs")
    for idx, subq in enumerate(sub_questions):
        with st.expander(f"**Graphing Sub-question {idx+1}:** {subq}", expanded=True):
            try:
                st.write("➡️ **Step A: Generating SQL Query**")
                # ... (SQL generation code is unchanged) ...
                sql_output = sql_generation_chain.invoke({"general_context": general_context, "graph_question": subq}).strip()
                sql_query = extract_sql(sql_output)
                st.code(sql_query, language="sql")

                st.write("➡️ **Step B: Executing Query and Plotting Graph**")
                with st.spinner("Fetching data and building graph..."):
                    df = pd.read_sql(sql_query, db._engine)
                    
                    if not df.empty and 'timestamp' in df.columns and 'value' in df.columns:
                        st.dataframe(df.head())
                        df['timestamp'] = pd.to_datetime(df['timestamp'])

                        # === Fix starts here ===
                        # 1. Create a base figure for the Streamlit UI display
                        # This version is kept simple to ensure it looks good in the UI.
                        fig_for_display = px.line(df, x='timestamp', y='value', title=subq)
                        # A better time format for clarity across days
                        fig_for_display.update_xaxes(tickformat='%b %d, %H:%M')
                        
                        # Display the clean figure in Streamlit
                        st.plotly_chart(fig_for_display, use_container_width=True)

                        # 2. Prepare a separate, enhanced figure for the HTML export
                        # We can apply more detailed styling here without affecting the UI
                        fig_for_export = px.line(df, x='timestamp', y='value') # Re-create for a clean slate
                        wrapped_title = "<br>".join(textwrap.wrap(subq, width=80))
                        fig_for_export.update_layout(
                            title_text=wrapped_title,
                            title_x=0.5,
                            xaxis_title='Time',
                            yaxis_title='Value'
                        )
                        fig_for_export.update_xaxes(tickformat='%b %d, %H:%M')

                        # 3. Add the *export-ready* figure to our list
                        figures_to_export.append((subq, fig_for_export))
                        # === Fix ends here ===

                    else:
                        st.warning("Query executed, but the result is not graphable.")
                        st.dataframe(df)

            except Exception as e:
                st.error(f"An error occurred while processing this sub-question: {e}")

    # --- 7. Export Section (Interactive HTML Report) ---
    if figures_to_export:
        st.subheader("3. Export Results")
        with st.spinner("Generating interactive HTML report..."):
            html_parts = []
            report_title = f"HVAC Graph Report - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}"
            
            html_header = f"""
            <html><head><title>{report_title}</title>
            <style>
                body {{ font-family: sans-serif; margin: 40px; background-color: #f0f2f6; }}
                h1 {{ text-align: center; color: #31333F; }}
                .graph-container {{ border: 1px solid #ddd; border-radius: 10px; margin-bottom: 40px; padding: 20px; background-color: #fff; box-shadow: 0 4px 8px rgba(0,0,0,0.1);}}
                h2 {{ font-size: 1.2em; color: #31333F; border-bottom: 2px solid #0068c9; padding-bottom: 10px; }}
            </style></head><body><h1>{report_title}</h1>
            """
            html_parts.append(html_header)

            for caption, fig in figures_to_export:
                graph_html = fig.to_html(full_html=False, include_plotlyjs='cdn')
                container_html = f"""
                <div class="graph-container"><h2>{caption}</h2>{graph_html}</div>
                """
                html_parts.append(container_html)
            
            html_footer = "</body></html>"
            html_parts.append(html_footer)
            final_html = "\n".join(html_parts)
            
        st.download_button(
            label="📥 Download Interactive HTML Report",
            data=final_html,
            file_name=f"hvac_interactive_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
            mime="text/html"
        )