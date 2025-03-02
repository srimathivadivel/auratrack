import streamlit as st
import pandas as pd
import plotly.express as px

def show_dashboard():  # Removed username parameter
    st.header("Social Media Data Dashboard")
    uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("### Data Preview", df.head())
        
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        if numeric_cols.any():
            col = st.selectbox("Select a column for histogram", numeric_cols)
            fig = px.histogram(df, x=col, title=f"Distribution of {col}")
            st.plotly_chart(fig)
        else:
            st.warning("No numeric columns found for visualization.")