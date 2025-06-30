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
import markdown
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
        # Add all your relevant tables here
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
# *** UPDATED: A simpler, more robust, example-driven Decomposer Prompt ***
DECOMPOSER_PROMPT_TEMPLATE = """
You are an expert AI that breaks down a complex user question into a list of simple, self-contained, graphable questions.
Your primary goal is to accurately parse all metrics, all entities, and especially time frames.
Use the **Current Date** to resolve relative time expressions like 'yesterday' or 'last month'.
Your output MUST be ONLY a valid JSON array of strings. Follow the format and logic shown in the examples precisely.

---
**Current Date:** {current_date}
---
**EXAMPLES:**

**Example 1: A specific single day and multiple entities/metrics.**
* **User Question:** "What is the trend of return air temperature and supply air temperature for AHU 005ca0 and 005da0 for January 12 2024?"
* **Reasoning:** The user specified a single, absolute date: "January 12 2024". I will use the format "on 2024-01-12". The query involves two metrics (return air temperature, supply air temperature) and two entities (AHU 005ca0, AHU 005da0), so I must generate 2x2=4 sub-questions.
* **JSON Output:**
    [
        "What was the trend of Return Air Temperature for ahu_005ca0_om_p on 2024-01-12?",
        "What was the trend of Supply Air Temperature for ahu_005ca0_om_p on 2024-01-12?",
        "What was the trend of Return Air Temperature for ahu_005da0_om_p on 2024-01-12?",
        "What was the trend of Supply Air Temperature for ahu_005da0_om_p on 2024-01-12?"
    ]

**Example 2: A relative time period.**
* **User Question:** "Show me a trend of the Return Air Temperature for AHU 5da0 last month."
* **Reasoning:** The user asked for "last month". The current date is {current_date}. The previous full month was May 2025. Therefore, the date range is "from 2025-05-01 to 2025-05-31". I will generate one sub-question with this date range.
* **JSON Output:**
    [
        "What was the trend of Return Air Temperature for ahu_005da0_om_p for the date range from 2025-05-01 to 2025-05-31?"
    ]

**Example 3: A relative single day.**
* **User Question:** "What was the Supply Air Temperature for ahu_005ea0_om_p yesterday?"
* **Reasoning:** The user asked for "yesterday". The current date is {current_date}. Yesterday was June 29, 2025. I will use the format "on 2025-06-29".
* **JSON Output:**
    [
        "What was the Supply Air Temperature for ahu_005ea0_om_p on 2025-06-29?"
    ]
---
**YOUR TASK:**

**Current Date:** {current_date}
**User Question:** "{complex_question}"
**Reasoning:**
**JSON Output:**
"""
# Note: We still only need complex_question and current_date as inputs. The model will generate the reasoning internally.
decomposer_chain = PromptTemplate(input_variables=["complex_question", "current_date"], template=DECOMPOSER_PROMPT_TEMPLATE) | llm


# SQL and Analysis Prompts (Unchanged)
GRAPH_SQL_PROMPT_TEMPLATE = """
You are an expert data analyst who writes syntactically correct MySQL queries for generating time-series line graphs. Your SOLE purpose is to write a single MySQL query that retrieves data suitable for a line chart based on the user's question and the provided schema.
**CRITICAL RULES:**
1. **NEVER Invent Columns:** You must only use columns that explicitly exist in the table schema provided. Do not hallucinate column names like `timestamp` or `RAT`.
2. **ALWAYS ALIAS COLUMNS:** The final query **MUST** select exactly two columns with specific aliases: - The first column must be the time column (e.g., `measured_time`) and **MUST** be aliased as `timestamp`. - The second column must be the numeric data column and **MUST** be aliased as `value`.
3. **KEY-VALUE TABLE LOGIC:** For key-value tables like `ahu_..._om_p`, the metric (e.g., 'Return Air Temperature') corresponds to a `param_id`. To get its value, you **MUST**: - Add a `WHERE param_id = '...'` clause. - `SELECT` the `param_value` column and alias it `AS value`.
4. **SORT BY TIME:** The results **MUST** be sorted by the timestamp in ascending order (`ORDER BY timestamp ASC`).
5. **SQL ONLY:** Your output **MUST** be ONLY the raw MySQL query. No explanations or markdown.
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

ANALYSIS_PROMPT_TEMPLATE = """
You are a data analyst with general knowledge of building systems like HVAC. Your task is to analyze a given time-series dataset. Your Goal: 1.  Identify basic, visible patterns in the provided data columns (`timestamp`, `value`). 2.  Based on your general knowledge, suggest what these patterns *COULD* hypothetically mean for the specific metric in the user's query. 3.  You MUST use speculative and cautious language.
**CRITICAL RULES:**
- **Your analysis MUST be strictly grounded in the provided data. YOU MUST ONLY DISCUSS THE METRIC MENTIONED IN THE USER'S QUERY (represented in the 'value' column).**
- **DO NOT mention or speculate about other metrics like humidity, pressure, fan speed, etc., unless they are explicitly part of the user's query.**
- Your entire response MUST be ONLY the analysis itself.
- DO NOT include conversational openings or closings (e.g., "Based on the data...", "Overall...").
- DO NOT ask follow-up questions or suggest next steps.
- DO NOT invent specific numbers, setpoints, or thresholds.
- USE PHRASES LIKE: "This might suggest...", "A possible reason could be...", "The data shows a pattern of...".
- Keep your analysis to 2-3 sentences.
---
**Context: The data represents the following user query:**
{graph_question}
**Time-Series Data (in CSV format with 'timestamp' and 'value' columns):**
{time_series_data}
---
**Concise, Grounded Analysis Paragraph:**
"""
analysis_chain = PromptTemplate(input_variables=["graph_question", "time_series_data"], template=ANALYSIS_PROMPT_TEMPLATE) | llm


# --- 5. SQL Extraction Helper Function (Unchanged) ---
def _looks_like_sql(text: str) -> bool:
    return any(text.strip().upper().startswith(kw) for kw in ["SELECT", "WITH"])

def extract_sql(llm_output: str) -> str:
    llm_output = llm_output.strip()
    match = re.search(r"```sql\s*(.*?)\s*```", llm_output, re.DOTALL | re.IGNORECASE)
    if match: return match.group(1).strip()
    match = re.search(r"```(.*?)```", llm_output, re.DOTALL)
    if match:
        potential_sql = match.group(1).strip()
        if _looks_like_sql(potential_sql): return potential_sql
    if _looks_like_sql(llm_output): return llm_output
    return llm_output

# --- 6. Streamlit UI (Unchanged) ---
st.title("📊 HVAC Analysis & Graphing Assistant")
st.markdown("Ask for a time-series trend and the AI will generate the corresponding graphs with analysis.")

question = st.text_area("Ask a graphable question:", placeholder="Trend of return air temp for AHU 005ca0 last week")

if st.button("Generate Report") and question:
    if not db or not llm:
        st.error("Database or LLM not initialized. Please check your connection.")
        st.stop()
    
    st.subheader("1. Decomposing Question")
    with st.spinner("Breaking down question into specific time frames..."):
        current_date_str = datetime.datetime.now().strftime('%Y-%m-%d')
        # We pass the current date twice, once for the context display and once for the actual task block
        decomposer_input = {"complex_question": question, "current_date": current_date_str}
        decomposed_raw = decomposer_chain.invoke(decomposer_input)
        try:
            # The model might output the reasoning text before the JSON. We need to find the JSON block.
            json_match = re.search(r'\[.*\]', decomposed_raw, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                sub_questions = json.loads(json_str)
            else:
                 st.warning("Could not find a valid JSON array in the output. Using original question.")
                 sub_questions = [question]
        except (json.JSONDecodeError, AttributeError, TypeError):
            st.warning("Could not decompose the question. Using the original question as a single task.")
            sub_questions = [question]
    st.json(sub_questions)
    
    report_items = []

    st.subheader("2. Generating Graphs & Analysis")
    for idx, subq in enumerate(sub_questions):
        with st.expander(f"**Analysis for:** {subq}", expanded=True):
            try:
                st.write("➡️ **Generating SQL Query...**")
                sql_output = sql_generation_chain.invoke({"general_context": general_context, "graph_question": subq}).strip()
                sql_query = extract_sql(sql_output)
                st.code(sql_query, language="sql")

                st.write("➡️ **Fetching Data & Plotting Graph...**")
                df = pd.read_sql(sql_query, db._engine)
                
                if df.empty or 'timestamp' not in df.columns or 'value' not in df.columns:
                    st.warning("Query executed, but the result is not graphable.")
                    if not df.empty: st.dataframe(df)
                    continue

                df['timestamp'] = pd.to_datetime(df['timestamp'])
                
                fig_for_display = px.line(df, x='timestamp', y='value', title=subq, render_mode='svg')
                fig_for_display.update_xaxes(tickformat='%b %d, %H:%M')
                st.plotly_chart(fig_for_display, use_container_width=True)

                st.write("➡️ **Analyzing Data...**")
                analysis_text = "No analysis generated."
                with st.spinner("AI is performing a general analysis..."):
                    data_csv = df.to_csv(index=False)
                    analysis_input = {"graph_question": subq, "time_series_data": data_csv}
                    analysis_text = analysis_chain.invoke(analysis_input)
                
                st.info(analysis_text)

                fig_for_export = px.line(df, x='timestamp', y='value', render_mode='svg')
                wrapped_title = "<br>".join(textwrap.wrap(subq, width=80))
                fig_for_export.update_layout(title_text=wrapped_title, title_x=0.5, xaxis_title='Time', yaxis_title='Value')
                fig_for_export.update_xaxes(tickformat='%b %d, %H:%M')
                
                report_items.append((subq, fig_for_export, analysis_text))

            except Exception as e:
                st.error(f"An error occurred while processing this sub-question: {e}")

    # --- 7. Export Section ---
    if report_items:
        st.subheader("3. Export Full Report")
        with st.spinner("Generating interactive HTML report..."):
            html_parts = []
            report_title = f"HVAC Analysis Report - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}"
            
            html_header = f"""
            <html><head><title>{report_title}</title>
            <style>
                body {{ font-family: sans-serif; margin: 40px; background-color: #f0f2f6; }}
                h1 {{ text-align: center; color: #31333F; }}
                .item-container {{ border: 1px solid #ddd; border-radius: 10px; margin-bottom: 40px; padding: 20px; background-color: #fff; box-shadow: 0 4px 8px rgba(0,0,0,0.1);}}
                h2 {{ font-size: 1.2em; color: #31333F; border-bottom: 2px solid #0068c9; padding-bottom: 10px; }}
                .analysis {{ font-style: italic; color: #333; background-color: #eaf3fb; border-left: 5px solid #0068c9; padding: 15px; margin-top: 20px; border-radius: 5px; }}
                .analysis strong {{ font-style: normal; }}
            </style></head><body><h1>{report_title}</h1>
            """
            html_parts.append(html_header)

            for caption, fig, analysis in report_items:
                analysis_html = markdown.markdown(analysis)
                graph_html = fig.to_html(full_html=False, include_plotlyjs='cdn')
                container_html = f"""
                <div class="item-container">
                    <h2>{caption}</h2>
                    {graph_html}
                    <div class="analysis">
                        <strong>Analysis:</strong>
                        {analysis_html}
                    </div>
                </div>
                """
                html_parts.append(container_html)
            
            html_footer = "</body></html>"
            html_parts.append(html_footer)
            final_html = "\n".join(html_parts)
            
        st.download_button(
            label="📥 Download Full Interactive Report",
            data=final_html,
            file_name=f"hvac_analysis_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
            mime="text/html"
        )
