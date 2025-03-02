import streamlit as st
import pandas as pd
import plotly.express as px
from db import save_user_data, get_user_data
from datetime import datetime

def show_tracker():
    st.header("Daily Social Media Tracker")
    
    with st.form("tracker_form"):
        # Date selector with default to today
        date = st.date_input("Date", datetime.now())
        
        # Core fields (matching existing database structure)
        screentime = st.number_input("Screentime (hours)", min_value=0.0, step=0.1)
        
        # Platform specific tracking
        st.subheader("Platform Details (Optional)")
        platforms = st.multiselect(
            "Platforms used today", 
            ["Instagram", "TikTok", "Facebook", "Twitter/X", "YouTube", "LinkedIn", "Reddit", "Snapchat", "Other"]
        )
        
        # Platform times as a text field to avoid database structure changes
        platform_details = ""
        if platforms:
            platform_times = {}
            for platform in platforms:
                minutes = st.number_input(f"{platform} time (minutes)", min_value=0, step=5)
                if minutes > 0:
                    platform_times[platform] = minutes
            
            if platform_times:
                platform_details = ", ".join([f"{p}: {t}min" for p, t in platform_times.items()])
        
        # Emotional state tracking
        st.subheader("Emotional State")
        
        # Primary emotion - using existing field
        emotion = st.selectbox(
            "Primary emotion today", 
            ["Happy", "Sad", "Anxious", "Excited", "Angry", "Neutral", "Content", "Frustrated", "Overwhelmed", "Peaceful"]
        )
        
        emotion_notes = []
        
        # More nuanced emotional state using slider
        if emotion in ["Happy", "Excited", "Content", "Peaceful"]:
            emotion_intensity = st.slider("Intensity of positive feeling", 1, 10, 5)
            emotion_notes.append(f"Emotion intensity: {emotion_intensity}/10")
        elif emotion in ["Sad", "Anxious", "Angry", "Frustrated", "Overwhelmed"]:
            emotion_intensity = st.slider("Intensity of negative feeling", 1, 10, 5)
            emotion_notes.append(f"Emotion intensity: {emotion_intensity}/10")
        
        # Before/after emotional state
        emotion_before = st.select_slider(
            "How did you feel BEFORE using social media today?",
            options=["Terrible", "Bad", "Somewhat Bad", "Neutral", "Somewhat Good", "Good", "Great"]
        )
        emotion_notes.append(f"Before: {emotion_before}")
        
        emotion_after = st.select_slider(
            "How do you feel AFTER using social media today?",
            options=["Terrible", "Bad", "Somewhat Bad", "Neutral", "Somewhat Good", "Good", "Great"]
        )
        emotion_notes.append(f"After: {emotion_after}")
        
        # Content interactions
        st.subheader("Content Interactions")
        
        content_type = st.multiselect(
            "What type of content did you engage with today?",
            ["News", "Entertainment", "Educational", "Friends/Family Updates", "Influencer Content", 
             "Political", "Inspirational", "Hobby-related", "Work-related", "Other"]
        )
        if content_type:
            emotion_notes.append(f"Content: {', '.join(content_type)}")
        
        interaction_type = st.multiselect(
            "How did you interact with content?",
            ["Passive Scrolling", "Liking/Reacting", "Commenting", "Sharing", "Creating Posts", 
             "Direct Messaging", "Video Calls", "Other"]
        )
        if interaction_type:
            emotion_notes.append(f"Interaction: {', '.join(interaction_type)}")
        
        # Impact assessment
        st.subheader("Impact Assessment")
        
        productivity_impact = st.select_slider(
            "Impact on your productivity today",
            options=["Very Negative", "Negative", "Slightly Negative", "Neutral", "Slightly Positive", "Positive", "Very Positive"]
        )
        emotion_notes.append(f"Productivity impact: {productivity_impact}")
        
        focus_impact = st.select_slider(
            "Impact on your focus/concentration today",
            options=["Very Negative", "Negative", "Slightly Negative", "Neutral", "Slightly Positive", "Positive", "Very Positive"]
        )
        emotion_notes.append(f"Focus impact: {focus_impact}")
        
        # User notes
        user_notes = st.text_area("Additional notes or observations")
        
        # Combine all notes
        all_notes = []
        if platform_details:
            all_notes.append(f"Platform details: {platform_details}")
        
        all_notes.extend(emotion_notes)
        
        if user_notes:
            all_notes.append(f"User notes: {user_notes}")
            
        notes = "\n".join(all_notes)
        
        # Submit button
        submit = st.form_submit_button("Log Entry")
        
        if submit:
            # Format date as string for storage
            date_str = date.strftime("%Y-%m-%d")
            
            # Save to database (using existing structure)
            save_user_data("guest", date_str, screentime, emotion, notes)
            st.success("Entry logged successfully!")

    df = get_user_data("guest")
    if not df.empty:
        st.subheader("Your Data Insights")
        
        with st.expander("View Raw Data"):
            st.dataframe(df)
        
        tab1, tab2 = st.tabs(["Usage Trends", "Emotion Analysis"])
        
        with tab1:
            if "date" in df.columns and "screentime" in df.columns:
                try:
                    if not pd.api.types.is_datetime64_dtype(df["date"]):
                        df["date"] = pd.to_datetime(df["date"])
                    
                    fig1 = px.line(df, x="date", y="screentime", title="Screentime Trend")
                    st.plotly_chart(fig1)
                except Exception as e:
                    st.error(f"Error creating screentime chart: {e}")
        
        with tab2:
            # Emotion distribution
            if "emotion" in df.columns:
                try:
                    emotion_counts = df['emotion'].value_counts().reset_index()
                    emotion_counts.columns = ["emotion", "count"]
                    fig2 = px.pie(emotion_counts, values="count", names="emotion", title="Emotion Distribution")
                    st.plotly_chart(fig2)
                except Exception as e:
                    st.error(f"Error creating emotion chart: {e}")
                
                try:
                    def extract_emotion_value(note_text, keyword):
                        if pd.isna(note_text):
                            return None
                        
                        emotion_map = {
                            "Terrible": 1, "Bad": 2, "Somewhat Bad": 3, 
                            "Neutral": 4, 
                            "Somewhat Good": 5, "Good": 6, "Great": 7
                        }
                        
                        for line in note_text.split('\n'):
                            if line.startswith(f"{keyword}:"):
                                for emotion, value in emotion_map.items():
                                    if emotion in line:
                                        return value
                        return None
                    
                    df["before_value"] = df["notes"].apply(lambda x: extract_emotion_value(x, "Before"))
                    df["after_value"] = df["notes"].apply(lambda x: extract_emotion_value(x, "After"))
                    
                    df["emotion_change"] = df["after_value"] - df["before_value"]
                    
                    emotion_change_df = df.dropna(subset=["emotion_change"])
                    
                    if not emotion_change_df.empty:
                        fig_emotion = px.bar(emotion_change_df, x="date", y="emotion_change", 
                                            title="Emotional Change After Social Media Use",
                                            color="emotion_change",
                                            color_continuous_scale=["red", "yellow", "green"])
                        st.plotly_chart(fig_emotion)
                except Exception as e:
                    st.info("Advanced emotion analysis will be available after more data is collected with the new form.")