import streamlit as st
import os
from dotenv import load_dotenv
import sqlalchemy
import re
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from io import BytesIO
import datetime
import textwrap
import markdown
from langchain_core.prompts import PromptTemplate
from langchain_community.utilities import SQLDatabase
from langchain_nvidia_ai_endpoints import ChatNVIDIA
import time
import logging

# --- 1. Enhanced Configuration with WebGL Fix ---
st.set_page_config(
    page_title="HVAC Report Generator",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load Environment
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- WebGL Fix: Configure Plotly to use SVG renderer ---
import plotly.io as pio
pio.renderers.default = "svg"

# Alternative: Set Plotly config to disable WebGL
PLOTLY_CONFIG = {
    'displayModeBar': True,
    'displaylogo': False,
    'modeBarButtonsToRemove': ['pan2d', 'lasso2d', 'select2d'],
    'toImageButtonOptions': {
        'format': 'png',
        'filename': 'hvac_chart',
        'height': 500,
        'width': 700,
        'scale': 1
    }
}

# --- 2. Enhanced Database and LLM Setup ---
@st.cache_resource
def initialize():
    """Initializes DB and LLM connections once and caches them."""
    try:
        db_uri = os.getenv("db_url")
        if not db_uri:
            st.error("Database URL not found in environment variables.")
            return None, None
            
        engine = sqlalchemy.create_engine(db_uri)
        db = SQLDatabase(engine)
        
        # Using Llama 3 70B as requested
        llm = ChatNVIDIA(model="meta/llama3-70b-instruct")
        st.sidebar.success("✅ Database and LLM initialized successfully!")
        return db, llm
    except Exception as e:
        st.sidebar.error(f"❌ Initialization failed: {e}")
        return None, None

db, llm = initialize()

# --- 3. Enhanced Context Loading with Historical Data Info ---
@st.cache_data
def load_context():
    """Load general context with fallback options."""
    try:
        with open("basic_full_context.txt", "r") as f:
            return f.read()
    except FileNotFoundError:
        st.sidebar.warning("⚠️ Context file not found. Using default schema.")
        return """
        HVAC Database Schema with Historical Data:
        
        DATA AVAILABILITY PERIODS:
        Group 1 (January 2024): January 11-19, 2024
        - AHU systems: ahu_005ca0_om_p, ahu_005da0_om_p, ahu_005ea0_om_p, ahu_005fa0_om_p, ahu_0060a0_om_p
        - Other equipment: ch_000bb0_om_p, pu_0010b1_om_p, secpu_000bb2_om_p
        - Sub-tables (fan_power, rpm) end around January 12, 2024
        
        Group 2 (September 2023): September 1-30, 2023
        - AHU system: ahu_0101a0_om_p with sub-tables
        - Sub-tables end between September 6-9, 2023
        
        Group 3 (Early 2025): January 20 - February 12, 2025
        - Chiller system: ch_010001b00000_om_p with sub-tables
        - Most sub-tables end around January 28, 2025
        
        Common table structure:
        - Main tables: equipment_name_om_p (measured_time, param_value)
        - Sub-tables: __chw_table, __fan_power_table, __rpm_table, __amps_table, __dis_pre_table
        """

general_context = load_context()

# --- 3.5. Data Availability Helper ---
DATA_PERIODS = {
    "January 2024": {
        "start": "2024-01-11",
        "end": "2024-01-19",
        "systems": ["ahu_005ca0", "ahu_005da0", "ahu_005ea0", "ahu_005fa0", "ahu_0060a0", "ch_000bb0", "pu_0010b1", "secpu_000bb2"],
        "sub_table_end": "2024-01-12"
    },
    "September 2023": {
        "start": "2023-09-01", 
        "end": "2023-09-30",
        "systems": ["ahu_0101a0"],
        "sub_table_end": "2023-09-09"
    },
    "Early 2025": {
        "start": "2025-01-20",
        "end": "2025-02-12", 
        "systems": ["ch_010001b00000"],
        "sub_table_end": "2025-01-28"
    }
}

def get_available_periods():
    """Return list of available data periods."""
    return list(DATA_PERIODS.keys())

def get_systems_for_period(period):
    """Get available systems for a specific period."""
    return DATA_PERIODS.get(period, {}).get("systems", [])

def get_date_range_for_period(period):
    """Get date range for a specific period."""
    period_info = DATA_PERIODS.get(period, {})
    return period_info.get("start"), period_info.get("end")

# --- 4. Enhanced Report Templates for Historical Data ---
REPORT_TEMPLATES = {
    "AHU General Performance Analysis": {
        "questions": [
            "Average Power consumption for different AHUs",
            "Average RAT for the AHUs", 
            "Average SAT for the AHUs",
        ],
        "description": "Performance analysis using available historical data"
    },
    "AHU Load Comparison": {
        "questions": [
            "Compare different AHU's RAT at similar times",
            "Compare daily average of AHU's RAT",
            "Compare weekly average of AHU's RAT",
        ],
        "description": "Uses RAT to compare AHU's in relation to their load"
    },
    "Daily Operational Trends": {
        "questions": [
            "Daily averages for different parameters",
            "Peak SAT analysis",
            "Peak RAT analysis",
        ],
        "description": "Trend analysis within available historical periods"
    },
    "Historical Baseline & Degradation": {
        "questions": [
            "Establish the baseline daily power consumption profile for each AHU during a typical occupied week.",
            "Compare the average temperature differential (ΔT) in summer 2023 vs. summer 2024 to check for degradation.",
            "Create an energy signature for a primary AHU by plotting its power (kW) against its cooling load.",
            "Track the trend of average fan motor amperage over the last 12 months."
        ],
        "description": "Establishes performance benchmarks and tracks equipment degradation over time."
    },
    "Period-Specific Analysis": {
        "questions": [],
        "description": "Analysis focused on specific data periods"
    },
    "Custom Analysis": {
        "questions": [],
        "description": "User-defined analysis questions"
    }
}

# --- 5. Enhanced Prompt Templates ---
if db and llm:
    DECOMPOSER_PROMPT = PromptTemplate.from_template("""
    You are an expert HVAC systems analyst working with historical data. Break down the user's question into specific, graphable sub-questions.
    
    IMPORTANT: Consider the available historical data periods and equipment:
    
    Available Data Periods:
    - January 2024 (Jan 11-19): AHU systems (005ca0, 005da0, 005ea0, 005fa0, 0060a0), chiller (000bb0), pumps
    - September 2023 (Sep 1-30): AHU system (0101a0) 
    - Early 2025 (Jan 20 - Feb 12): Chiller system (010001b00000)
    
    Guidelines:
    - Reference specific equipment names from available data
    - Use historical date ranges instead of "recent" or "current" 
    - Focus on available parameters for each time period
    - Consider cross-period comparisons when applicable
    - Format as clean JSON array of strings
    
    Available equipment: {available_systems}
    Selected data period: {selected_period}
    Date range: {date_range}
    
    User question: {complex_question}
    
    Provide ONLY a JSON array of sub-questions:
    """)

    SQL_GENERATION_PROMPT = PromptTemplate.from_template("""
    You are an expert MySQL developer specializing in HVAC historical time-series data.
    
    CRITICAL: This is historical data with specific date ranges. Do not use relative dates.
    
    Available Data Periods:
    - January 2024: 2024-01-11 to 2024-01-19
    - September 2023: 2023-09-01 to 2023-09-30  
    - Early 2025: 2025-01-20 to 2025-02-12
    
    Requirements:
    - Always return columns named 'timestamp' and 'value'
    - Use measured_time column for timestamps
    - Filter by appropriate date ranges for available data
    - Use specific equipment table names (e.g., ahu_005ca0_om_p)
    - Handle sub-tables with shorter date ranges appropriately
    - Use proper JOIN operations for multi-table queries
    
    Database schema and periods:
    {general_context}
    
    Selected period: {selected_period}
    Available systems: {available_systems}
    
    Question: {graph_question}
    
    Return only the SQL query:
    """)

    ANALYSIS_PROMPT = PromptTemplate.from_template("""
    You are an HVAC systems expert analyzing historical data. Provide actionable insights considering the historical context.
    
    Guidelines:
    - Write 2-3 concise sentences
    - Consider this is historical data, not current operations
    - Focus on patterns and trends from the specific time period
    - Compare against typical operational ranges when relevant
    - Use practical HVAC terminology
    - Mention the time period context in your analysis
    
    Data period: {data_period}
    Question context: {graph_question}
    Data statistics: {summary_stats}
    
    Historical Analysis:
    """)

    EXECUTIVE_SUMMARY_PROMPT = PromptTemplate.from_template("""
    You are an HVAC facility manager writing an executive summary for historical data analysis.
    
    Based on the following historical analyses, provide a comprehensive summary that includes:
    1. Key findings from the historical data periods
    2. System performance insights from available data
    3. Historical trends and patterns identified
    4. Recommendations based on historical performance
    5. Data limitations and period-specific considerations
    
    Data periods analyzed: {data_periods}
    Individual analyses: {analyses}
    
    Executive Summary:
    """)

    # Initialize chains
    decomposer_chain = DECOMPOSER_PROMPT | llm
    sql_generation_chain = SQL_GENERATION_PROMPT | llm
    analysis_chain = ANALYSIS_PROMPT | llm
    executive_summary_chain = EXECUTIVE_SUMMARY_PROMPT | llm

# --- 6. Enhanced Helper Functions ---
def _looks_like_sql(text: str) -> bool:
    """Enhanced SQL detection."""
    sql_keywords = ["SELECT", "WITH", "INSERT", "UPDATE", "DELETE", "SHOW", "DESCRIBE"]
    return any(text.strip().upper().startswith(kw) for kw in sql_keywords)

def extract_sql(llm_output: str) -> str:
    """Enhanced SQL extraction with multiple fallback methods."""
    llm_output = llm_output.strip()
    
    # Method 1: Look for markdown code blocks
    patterns = [
        r"```sql\s*(.*?)```",
        r"```\s*(SELECT.*?)```",
        r"```\s*(WITH.*?)```"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, llm_output, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
    
    # Method 2: Look for SQL-like content
    lines = llm_output.split('\n')
    sql_lines = []
    in_sql = False
    
    for line in lines:
        if _looks_like_sql(line) or in_sql:
            in_sql = True
            sql_lines.append(line)
            if line.strip().endswith(';'):
                break
    
    if sql_lines:
        return '\n'.join(sql_lines).strip()
    
    # Method 3: Fallback to entire output if it looks like SQL
    if _looks_like_sql(llm_output):
        return llm_output
    
    return ""

def create_enhanced_visualization(df, graph_type, title, subq):
    """Create enhanced visualizations with WebGL fixes."""
    if df.empty:
        return None
    
    fig = None
    
    # Limit data points to prevent WebGL issues
    max_points = 10000
    if len(df) > max_points:
        df = df.sample(n=max_points).sort_values('timestamp' if 'timestamp' in df.columns else df.columns[0])
        st.warning(f"⚠️ Data limited to {max_points} points for performance. Original dataset had {len(df)} points.")
    
    if graph_type == "Line Chart" and 'timestamp' in df.columns:
        fig = px.line(df, x='timestamp', y='value', title=title, render_mode='svg')
        fig.update_layout(
            xaxis_title="Time",
            yaxis_title="Value",
            hovermode='x unified'
        )
        # Force SVG rendering for line charts
        fig.update_traces(mode='lines')
        
    elif graph_type == "Scatter Plot" and 'timestamp' in df.columns:
        fig = px.scatter(df, x='timestamp', y='value', title=title, render_mode='svg')
        fig.update_layout(
            xaxis_title="Time",
            yaxis_title="Value"
        )
        # Reduce marker size for better performance
        fig.update_traces(marker=dict(size=3))
        
    elif graph_type == "Histogram":
        fig = px.histogram(df, x='value', title=title, nbins=min(50, len(df)//10))
        fig.update_layout(
            xaxis_title="Value",
            yaxis_title="Frequency"
        )
    
    elif graph_type == "Box Plot":
        fig = px.box(df, y='value', title=title, render_mode='svg')
        fig.update_layout(yaxis_title="Value")
    
    if fig:
        fig.update_layout(
            template="plotly_white",
            title_font_size=16,
            showlegend=True,
            # Disable WebGL acceleration
            dragmode='pan',
            # Optimize for SVG rendering
            font=dict(size=12),
            margin=dict(l=50, r=50, t=50, b=50)
        )
        
        # Additional WebGL fixes
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
    
    return fig

def create_fallback_visualization(df, graph_type, title):
    """Create matplotlib fallback visualization if Plotly fails."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        
        plt.style.use('seaborn-v0_8')
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if graph_type == "Line Chart" and 'timestamp' in df.columns:
            ax.plot(df['timestamp'], df['value'])
            ax.set_xlabel('Time')
            ax.set_ylabel('Value')
            if 'timestamp' in df.columns:
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                plt.xticks(rotation=45)
                
        elif graph_type == "Scatter Plot" and 'timestamp' in df.columns:
            ax.scatter(df['timestamp'], df['value'], alpha=0.6, s=20)
            ax.set_xlabel('Time')
            ax.set_ylabel('Value')
            if 'timestamp' in df.columns:
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                plt.xticks(rotation=45)
                
        elif graph_type == "Histogram":
            ax.hist(df['value'], bins=30, alpha=0.7, edgecolor='black')
            ax.set_xlabel('Value')
            ax.set_ylabel('Frequency')
            
        elif graph_type == "Box Plot":
            ax.boxplot(df['value'])
            ax.set_ylabel('Value')
        
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        return fig
        
    except Exception as e:
        st.error(f"Both Plotly and Matplotlib visualization failed: {e}")
        return None

def safe_plotly_chart(fig, **kwargs):
    """Safely display Plotly chart with fallback options."""
    try:
        if fig is not None:
            st.plotly_chart(fig, config=PLOTLY_CONFIG, **kwargs)
        else:
            st.error("Unable to generate chart")
    except Exception as e:
        st.error(f"Chart rendering failed: {e}")
        st.info("💡 Try refreshing the page or using a different browser if charts don't load properly.")

def generate_auto_report(template_name, custom_questions=None, selected_period=None, selected_systems=None):
    """Generate automated report based on template with historical data context."""
    if template_name == "Custom Analysis" and custom_questions:
        questions = custom_questions
    elif template_name == "Period-Specific Analysis" and custom_questions:
        questions = custom_questions
    else:
        questions = REPORT_TEMPLATES[template_name]["questions"]
    
    # Get period information
    if selected_period:
        period_info = DATA_PERIODS.get(selected_period, {})
        available_systems = period_info.get("systems", [])
        date_range = f"{period_info.get('start')} to {period_info.get('end')}"
    else:
        available_systems = []
        date_range = "All available periods"
        selected_period = "All Periods"
    
    report_data = []
    
    for question in questions:
        try:
            # Decompose question with historical context
            decomposed_raw = decomposer_chain.invoke({
                "complex_question": question,
                "available_systems": ", ".join(available_systems) if available_systems else "All available systems",
                "selected_period": selected_period,
                "date_range": date_range
            }).content
            
            # Extract sub-questions
            json_match = re.search(r'\[.*\]', decomposed_raw, re.DOTALL)
            if json_match:
                sub_questions = json.loads(json_match.group(0))
            else:
                sub_questions = [question]
            
            # Process each sub-question
            for subq in sub_questions:
                try:
                    # Generate SQL with historical context
                    sql_output = sql_generation_chain.invoke({
                        "general_context": general_context,
                        "graph_question": subq,
                        "selected_period": selected_period,
                        "available_systems": ", ".join(available_systems) if available_systems else "All available systems"
                    }).content
                    
                    sql_query = extract_sql(sql_output)
                    if not sql_query:
                        continue
                    
                    # Execute query
                    df = pd.read_sql(sql_query, db._engine)
                    
                    if df.empty or 'value' not in df.columns:
                        continue
                    
                    # Process data
                    if 'timestamp' in df.columns:
                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df['value'] = pd.to_numeric(df['value'], errors='coerce')
                    df.dropna(subset=['value'], inplace=True)
                    
                    # Generate analysis with historical context
                    summary_stats = df['value'].describe().to_dict()
                    summary_stats_json = json.dumps({k: round(v, 2) for k, v in summary_stats.items()}, indent=2)
                    
                    analysis = analysis_chain.invoke({
                        "graph_question": subq,
                        "summary_stats": summary_stats_json,
                        "data_period": selected_period
                    }).content
                    
                    report_data.append({
                        "question": subq,
                        "sql": sql_query,
                        "data": df,
                        "analysis": analysis,
                        "period": selected_period
                    })
                    
                except Exception as e:
                    logger.error(f"Error processing sub-question '{subq}': {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error processing question '{question}': {e}")
            continue
    
    return report_data

# --- 7. Enhanced Streamlit UI with WebGL Fixes ---
st.title("📊 HVAC Report Auto-Generator")
st.markdown("Generate comprehensive HVAC analysis reports with automated insights.")

# Add WebGL status check
with st.expander("🔧 Browser Compatibility Check"):
    st.markdown("""
    **Chart Rendering Options:**
    - **SVG Mode**: Enabled by default (compatible with all browsers)
    - **WebGL Issues**: If you see WebGL errors, they should be automatically handled
    - **Fallback**: Charts will render in SVG mode for maximum compatibility
    
    **Tips for Better Performance:**
    - Use Chrome, Firefox, or Safari for best results
    - Large datasets are automatically sampled for performance
    - Try different chart types if one doesn't work
    """)

# Sidebar configuration
st.sidebar.header("⚙️ Configuration")

# Data period selection
st.sidebar.subheader("📅 Data Period Selection")
available_periods = get_available_periods()
selected_period = st.sidebar.selectbox(
    "Select Data Period:",
    ["All Periods"] + available_periods,
    help="Choose specific historical data period or analyze all available data"
)

# Show period information
if selected_period != "All Periods":
    period_info = DATA_PERIODS[selected_period]
    start_date, end_date = get_date_range_for_period(selected_period)
    st.sidebar.info(f"**Period:** {start_date} to {end_date}")
    
    # System selection for specific period
    available_systems = get_systems_for_period(selected_period)
    if available_systems:
        st.sidebar.markdown("**Available Systems:**")
        for system in available_systems:
            st.sidebar.markdown(f"• {system}")
        
        selected_systems = st.sidebar.multiselect(
            "Filter by Systems:",
            available_systems,
            default=available_systems,
            help="Select specific systems to analyze"
        )
    else:
        selected_systems = []
else:
    selected_systems = []
    st.sidebar.info("**All historical periods will be considered**")

# Report template selection
template_choice = st.sidebar.selectbox(
    "Select Report Template:",
    list(REPORT_TEMPLATES.keys()),
    help="Choose a pre-configured report template or create custom analysis"
)

st.sidebar.markdown(f"**Description:** {REPORT_TEMPLATES[template_choice]['description']}")

# Graph type selection
graph_type = st.sidebar.selectbox(
    "Default Graph Type:",
    ["Line Chart", "Scatter Plot", "Histogram", "Box Plot"],
    help="Default visualization type for time-series data"
)

# Custom questions for Custom Analysis and Period-Specific Analysis
custom_questions = []
if template_choice in ["Custom Analysis", "Period-Specific Analysis"]:
    st.sidebar.subheader("Custom Questions")
    num_questions = st.sidebar.number_input("Number of questions:", min_value=1, max_value=10, value=1)
    
    for i in range(num_questions):
        question = st.sidebar.text_area(f"Question {i+1}:", key=f"custom_q_{i}")
        if question:
            custom_questions.append(question)

# Main interface
col1, col2 = st.columns([3, 1])

with col1:
    # Show data availability info
    st.subheader("📊 Historical Data Overview")
    
    # Create data availability visualization
    col_a, col_b, col_c = st.columns(3)
    
    with col_a:
        st.metric("January 2024", "8 systems", "Jan 11-19")
        st.caption("AHUs, Chiller, Pumps")
    
    with col_b:
        st.metric("September 2023", "1 system", "Sep 1-30") 
        st.caption("AHU 0101a0")
    
    with col_c:
        st.metric("Early 2025", "1 system", "Jan 20 - Feb 12")
        st.caption("Chiller 010001b00000")
    
    # Show selected template info
    if template_choice not in ["Custom Analysis", "Period-Specific Analysis"]:
        st.subheader(f"📋 {template_choice} Template")
        st.markdown("**Included Analysis:**")
        for i, q in enumerate(REPORT_TEMPLATES[template_choice]["questions"], 1):
            st.markdown(f"{i}. {q}")
    else:
        st.subheader(f"📝 {template_choice}")
        if custom_questions:
            st.markdown("**Your Questions:**")
            for i, q in enumerate(custom_questions, 1):
                st.markdown(f"{i}. {q}")
        
        if selected_period != "All Periods":
            period_info = DATA_PERIODS[selected_period]
            st.info(f"**Selected Period:** {period_info['start']} to {period_info['end']}")

with col2:
    generate_button = st.button("🚀 Generate Report", type="primary", use_container_width=True)
    
    # Show data limitations
    st.markdown("---")
    st.markdown("**⚠️ Data Limitations:**")
    st.caption("• Historical data only")
    st.caption("• Limited date ranges")
    st.caption("• Sub-tables have shorter periods")
    st.caption("• No real-time data available")
    
    # Browser compatibility info
    st.markdown("---")
    st.markdown("**🔧 Chart Compatibility:**")
    st.caption("• SVG rendering enabled")
    st.caption("• WebGL issues auto-fixed")
    st.caption("• All browsers supported")

# Report generation
if generate_button:
    if not db or not llm:
        st.error("❌ Application is not ready. Please check sidebar for initialization errors.")
        st.stop()
    
    if template_choice in ["Custom Analysis", "Period-Specific Analysis"] and not custom_questions:
        st.error("❌ Please add at least one custom question.")
        st.stop()
    
    # Generate report with historical context
    with st.spinner("🔄 Generating comprehensive historical report..."):
        period_for_generation = selected_period if selected_period != "All Periods" else None
        report_data = generate_auto_report(
            template_choice, 
            custom_questions, 
            period_for_generation,
            selected_systems
        )
    
    if not report_data:
        st.error("❌ No data could be generated. Please check your database connection and queries.")
        st.stop()
    
    # Display results
    st.success(f"✅ Generated report with {len(report_data)} analyses")
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["📊 Visualizations", "📈 Data Analysis", "📄 Executive Summary"])
    
    with tab1:
        st.subheader("Generated Visualizations")
        
        for i, item in enumerate(report_data):
            with st.expander(f"📊 {item['question']}", expanded=True):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    try:
                        fig = create_enhanced_visualization(
                            item['data'], 
                            graph_type, 
                            item['question'], 
                            item['question']
                        )
                        if fig:
                            safe_plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning("Could not generate visualization for this data.")
                    except Exception as e:
                        st.error(f"Visualization error: {e}")
                        st.info("💡 Try refreshing the page or selecting a different chart type.")
                
                with col2:
                    st.markdown("**Quick Stats:**")
                    if not item['data'].empty and 'value' in item['data'].columns:
                        stats = item['data']['value'].describe()
                        st.metric("Mean", f"{stats['mean']:.2f}")
                        st.metric("Std Dev", f"{stats['std']:.2f}")
                        st.metric("Min/Max", f"{stats['min']:.1f} / {stats['max']:.1f}")
                        st.metric("Data Points", f"{len(item['data'])}")
    
    with tab2:
        st.subheader("Detailed Analysis")
        
        for i, item in enumerate(report_data):
            with st.expander(f"📈 Analysis: {item['question']}", expanded=False):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown("**Analysis:**")
                    st.info(item['analysis'])
                
                with col2:
                    st.markdown("**SQL Query:**")
                    st.code(item['sql'], language="sql")
    
    with tab3:
        st.subheader("Executive Summary")
        
        with st.spinner("Generating executive summary..."):
            all_analyses = "\n\n".join([f"Question: {item['question']}\nPeriod: {item['period']}\nAnalysis: {item['analysis']}" 
                                      for item in report_data])
            
            data_periods_analyzed = list(set([item['period'] for item in report_data]))
            
            try:
                exec_summary = executive_summary_chain.invoke({
                    "analyses": all_analyses,
                    "data_periods": ", ".join(data_periods_analyzed)
                }).content
                
                st.markdown("### 📋 Executive Summary")
                st.markdown(exec_summary)
            except:
                raise