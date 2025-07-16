import os

# This script generates a SINGLE, simplified context file for a non-RAG approach.

# A curated list of the most important param_ids for the main AHU table
curated_ahu_params = {
    "OAT": "Outside Air Temperature.",
    "SAT": "Supply Air Temperature.",
    "RAT": "Return Air Temperature.",
    "MAT": "Mixed Air Temperature.",
    "SAT_SP": "Supply Air Temperature Setpoint.",
    "DSP": "Discharge Static Pressure.",
    "DSP_SP": "Setpoint for the discharge static pressure.",
    "SA_CFM": "Supply Air flow rate in cubic feet per minute.",
    "CHW_Vlv_Pos": "Position of the chilled water valve (0-100%).",
    "SAF_VFD_Speed": "Commanded speed to the Supply Air Fan VFD.",
    "RPM": "Revolutions per minute of a fan.",
    "FAN_POWER": "Measured power consumption of the fan."
}

# A curated list for the detailed chiller table
curated_chiller_params = {
    "CHW_In_Temp": "Chilled Water Inlet Temperature.",
    "CHW_Out_Temp": "Chilled Water Outlet Temperature.",
    "CH_Out_Temp_SP": "Chiller Outlet Temperature Setpoint.",
    "CDW_In_Temp": "Condenser Water Inlet Temperature.",
    "CDW_Out_Temp": "Condenser Water Outlet Temperature.",
    "Cmp_Dis_Pre": "Compressor Discharge Pressure.",
    "Cmp_Suc_Pre": "Compressor Suction Pressure.",
    "Dis_SuperHeat": "Discharge Superheat.",
    "amps": "Amperes (electrical current drawn).",
    "Fault_Status": "A flag indicating a system fault."
}

# The overall structure of the context file
# Using patterns to save space
context_definitions = [
    {
        'name': 'Primary AHU Data (Pattern: `ahu_[unit_code]_om_p`)',
        'desc': 'Key-value table for AHU sensor readings. Use `param_id` and `param_value` columns.',
        'columns': 'id, ss_id, measured_time, param_id, param_value, created_at, modified_at',
        'params': curated_ahu_params
    },
    {
        'name': 'AHU Sub-Tables (Patterns: `...__rpm_table`, `...__fan_power_table`, etc.)',
        'desc': 'These tables have traditional columns for specific AHU measurements like RPM and Fan Power.',
        'columns': 'measured_time, SA_CFM, DSP, RPM, FAN_POWER, FAN_POWER_Error, Fault_Status, etc.',
        'params': None
    },
    {
        'name': 'Detailed Chiller Data (Table: `ch_010001b00000_om_p`)',
        'desc': 'Key-value table for detailed chiller thermodynamics. Use `param_id` and `param_value`.',
        'columns': 'id, ss_id, measured_time, param_id, param_value, created_at, modified_at, measured_time_calculated',
        'params': curated_chiller_params
    },
    {
        'name': 'Chiller Modeling Tables (Pattern: `ch_...__[measurement]_table`)',
        'desc': 'Traditional tables for chiller fault detection, with columns like `Amps_Estimate`, `Amps_Error`, `Fault_Status`.',
        'columns': 'measured_time, CW_IPT, CW_OPT, CD_WET, amps, Disc_pres, Suct_pres, ..._Estimate, ..._Error, Fault_Status',
        'params': None
    },
    {
        'name': 'Pump Status Tables (e.g., `pu_0010b1_om_p`, `secpu_000bb2_om_p`)',
        'desc': 'Key-value tables for simple pump status.',
        'columns': 'id, ss_id, measured_time, param_id, param_value, created_at, modified_at',
        'params': { "Pri_Pmp_On_Off": "Primary Pump On/Off", "Pri_Pmp_Trip_SS": "Primary Pump Trip Status", "Sec_Pmp_On_Off": "Secondary Pump On/Off", "Sec_Pmp_Trip_SS": "Secondary Pump Trip Status" }
    }
]

# ==============================================================================
# 2. SCRIPT TO GENERATE THE SINGLE FILE
# ==============================================================================
output_filename = "basic_full_context.txt"

print(f"Generating single context file: '{output_filename}'...")

with open(output_filename, "w", encoding='utf-8') as f:
    f.write("--- DATABASE CONTEXT ---\n")
    f.write("This document provides essential context for all relevant tables in the database.\n\n")

    for context in context_definitions:
        f.write(f"### {context['name']}\n")
        f.write(f"- Description: {context['desc']}\n")
        f.write(f"- Columns: {context['columns']}\n")
        if context['params']:
            f.write("- Key Parameters (`param_id`):\n")
            for param, desc in context['params'].items():
                f.write(f"  - `{param}`: {desc}\n")
        f.write("\n")

print(f"Successfully wrote context to '{output_filename}'")
