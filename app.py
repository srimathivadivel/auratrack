import streamlit as st
from db import init_db, get_user_data, save_user_data
from dashboard import show_dashboard
from tracker import show_tracker

# Initialize SQLite database
init_db()

st.title("AuraTracker")

# Tabs for different features
tab1, tab2 = st.tabs(["CSV Dashboard", "Daily Tracker"])

with tab1:
    show_dashboard()  # No "guest" argument since show_dashboard takes no parameters
with tab2:
    show_tracker()    # Same for show_tracker