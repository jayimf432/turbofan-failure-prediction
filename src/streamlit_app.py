import streamlit as st
import requests
import pandas as pd
import json
import plotly.graph_objects as go

import os

# Configuration
API_URL = os.getenv("API_URL", "http://api:8000")

st.set_page_config(
    page_title="Turbofan Failure Prediction",
    page_icon="✈️",
    layout="wide"
)

st.title("✈️ Turbofan Engine Predictive Maintenance")
st.markdown("Predict engine failure probability based on sensor readings.")

# Sidebar for connection check
with st.sidebar:
    st.header("System Status")
    if st.button("Check API Health"):
        try:
            resp = requests.get(f"{API_URL}/health")
            if resp.status_code == 200:
                st.success(f"Online: {resp.json().get('timestamp')}")
            else:
                st.error("API Unhealthy")
        except Exception as e:
            st.error(f"Connection Failed: {e}")

# Main Input Form
st.header("Input Sensor Data")

with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        unit_number = st.number_input("Unit Number", min_value=1, value=1)
        time_in_cycles = st.number_input("Time in Cycles", min_value=1, value=100)
        setting_1 = st.number_input("Setting 1", value=-0.0007, format="%.4f")
        setting_2 = st.number_input("Setting 2", value=-0.0004, format="%.4f")
        setting_3 = st.number_input("Setting 3", value=100.0)
        
    with col2:
        s_2 = st.number_input("s_2 (Fan Inlet Temp)", value=641.82)
        s_3 = st.number_input("s_3 (LPC Outlet Temp)", value=1589.70)
        s_4 = st.number_input("s_4 (HPC Outlet Temp)", value=1400.60)
        s_7 = st.number_input("s_7 (HPC Pressure)", value=554.36)
        s_8 = st.number_input("s_8 (Fan Speed)", value=2388.06)
        s_9 = st.number_input("s_9 (Core Speed)", value=9046.19)
        
    with col3:
        s_11 = st.number_input("s_11 (Static Pressure)", value=47.47)
        s_12 = st.number_input("s_12 (Ratio)", value=521.66)
        s_13 = st.number_input("s_13 (Burner)", value=2388.02)
        s_14 = st.number_input("s_14 (HPC Outlet P)", value=8138.62)
        s_15 = st.number_input("s_15 (Bypass Ratio)", value=8.4195)
        s_17 = st.number_input("s_17 (HPC Loss)", value=392.0)
        s_20 = st.number_input("s_20 (Coolant Bleed)", value=39.06)
        s_21 = st.number_input("s_21 (LPT Coolant)", value=23.4190)

    # Hidden/Constant Sensors (Less important or constant)
    s_1 = 518.67
    s_5 = 14.62
    s_6 = 21.61
    s_10 = 1.30
    s_16 = 0.03
    s_18 = 2388
    s_19 = 100.0
    
    submit = st.form_submit_button("Predict Failure Risk")

if submit:
    # Construct Payload
    # NOTE: The API expects a LIST of history. 
    # For a simple demo, we pass just one point. 
    # The API will handle it (min_periods=1) but warn about "Short history".
    
    payload = {
        "data": [{
            "unit_number": unit_number,
            "time_in_cycles": time_in_cycles,
            "setting_1": setting_1, "setting_2": setting_2, "setting_3": setting_3,
            "s_1": s_1, "s_2": s_2, "s_3": s_3, "s_4": s_4, "s_5": s_5,
            "s_6": s_6, "s_7": s_7, "s_8": s_8, "s_9": s_9, "s_10": s_10,
            "s_11": s_11, "s_12": s_12, "s_13": s_13, "s_14": s_14, "s_15": s_15,
            "s_16": s_16, "s_17": s_17, "s_18": s_18, "s_19": s_19, "s_20": s_20,
            "s_21": s_21
        }]
    }
    
    try:
        with st.spinner("Analyzing Sensor Data..."):
            response = requests.post(f"{API_URL}/predict", json=payload)
            
        if response.status_code == 200:
            result = response.json()
            
            # Display Results
            st.divider()
            r_col1, r_col2 = st.columns([1, 2])
            
            with r_col1:
                risk = result['risk_level']
                color = "green" if risk == "Low" else "orange" if risk == "Medium" else "red"
                st.markdown(f"## Risk Level: :{color}[{risk}]")
                st.metric("Failure Probability", f"{result['failure_probability']:.2%}")
                
                if result['prediction'] == 1:
                    st.error("⚠️ FAILURE PREDICTED within 30 cycles")
                else:
                    st.success("✅ System Healthy")
                    
            with r_col2:
                st.subheader("Contributing Factors")
                factors = result.get('contributing_factors', [])
                if factors:
                    # Bar chart for SHAP
                    df_shap = pd.DataFrame(factors)
                    fig = go.Figure(go.Bar(
                        x=df_shap['impact'],
                        y=df_shap['feature'],
                        orientation='h'
                    ))
                    fig.update_layout(title="Feature Impact (SHAP Values)", autosize=False, height=300)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No explanation available.")
                    
            if result.get('warning'):
                st.warning(f"Note: {result['warning']}")
                
        else:
            st.error(f"Error {response.status_code}: {response.text}")
            
    except Exception as e:
        st.error(f"Failed to connect to API: {e}")
