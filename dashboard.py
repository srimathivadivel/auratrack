import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import re
from google.generativeai import GenerativeModel
import google.generativeai as genai
from dotenv import load_dotenv
import os



# Global variable for CSV file path - change this to match your file location
CSV_FILE_PATH = "Data/smmh.csv"

# Page configuration
st.set_page_config(
    layout="wide",
    page_title="AuraTracker",
    page_icon="🧠",
    initial_sidebar_state="expanded"
)

# Load environment variables from .env file
load_dotenv(dotenv_path="api.env")

# Theme-aware styling
def get_theme_specific_colors():
    # Check if theme is dark or light
    try:
        is_dark_theme = st.get_option("theme.base") == "dark"
    except:
        is_dark_theme = False  # Default to light if can't determine
   
    if is_dark_theme:
        # Dark theme colors
        header_color = "#BB86FC"
        subheader_color = "#03DAC6"
        text_color = "#FFFFFF"
        background_color = "#1E1E1E"
        box_background = "#2D2D2D"
        border_color = "#BB86FC"
        chart_colors = ["#BB86FC", "#03DAC6", "#CF6679", "#FFAB40", "#80D8FF", "#B388FF"]
        positive_color = "#03DAC6"   # Teal
        neutral_color = "#FFAB40"    # Amber
        negative_color = "#CF6679"   # Pink
        gradient_colors = ["#CF6679", "#FFAB40", "#03DAC6"]  # Negative to Positive
    else:
        # Light theme colors
        header_color = "#6200EE"
        subheader_color = "#3700B3"
        text_color = "#000000"
        background_color = "#FFFFFF"
        box_background = "#F5F5F5"
        border_color = "#6200EE"
        chart_colors = ["#6200EE", "#03DAC6", "#B00020", "#FF6D00", "#0091EA", "#7C4DFF"]
        positive_color = "#03DAC6"   # Teal
        neutral_color = "#FF6D00"    # Orange
        negative_color = "#B00020"   # Red
        gradient_colors = ["#B00020", "#FF6D00", "#03DAC6"]  # Negative to Positive
   
    return {
        "header_color": header_color,
        "subheader_color": subheader_color,
        "text_color": text_color,
        "background_color": background_color,
        "box_background": box_background,
        "border_color": border_color,
        "chart_colors": chart_colors,
        "positive_color": positive_color,
        "neutral_color": neutral_color,
        "negative_color": negative_color,
        "gradient_colors": gradient_colors,
    }

# Get theme colors
theme = get_theme_specific_colors()

# Apply custom styling with theme awareness
st.markdown(f"""
<style>
    .main-header {{
        font-size: 2.5rem;
        color: {theme["header_color"]};
        font-weight: 600;
        margin-bottom: 1rem;
    }}
    .sub-header {{
        font-size: 1.8rem;
        color: {theme["subheader_color"]};
        font-weight: 500;
        margin-top: 1rem;
    }}
    .insight-box {{
        background-color: {theme["box_background"]};
        padding: 1.2rem;
        border-radius: 8px;
        border-left: 5px solid {theme["border_color"]};
        margin: 1rem 0;
    }}
    .metric-container {{
        background-color: {theme["box_background"]};
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }}
    .metric-value {{
        font-size: 2rem;
        font-weight: 600;
        color: {theme["header_color"]};
    }}
    .metric-label {{
        font-size: 1rem;
        color: {theme["text_color"]};
        opacity: 0.8;
    }}
    .stTabs [data-baseweb="tab-list"] {{
        gap: 12px;
    }}
    .stTabs [data-baseweb="tab"] {{
        background-color: {theme["box_background"]};
        border-radius: 6px 6px 0px 0px;
        padding: 10px 16px;
        font-weight: 500;
    }}
    .stTabs [aria-selected="true"] {{
        background-color: {theme["box_background"]};
        border-bottom: 2px solid {theme["border_color"]};
    }}
    .st-emotion-cache-1y4p8pa {{
        max-width: 1200px;
    }}
    .score-badge {{
        display: inline-block;
        padding: 0.25em 0.6em;
        font-size: 0.9rem;
        font-weight: 700;
        line-height: 1;
        text-align: center;
        white-space: nowrap;
        vertical-align: baseline;
        border-radius: 0.25rem;
    }}
    .good {{
        background-color: {theme["positive_color"]};
        color: #fff;
    }}
    .moderate {{
        background-color: {theme["neutral_color"]};
        color: #fff;
    }}
    .poor {{
        background-color: {theme["negative_color"]};
        color: #fff;
    }}
</style>
""", unsafe_allow_html=True)

# Helper functions
def clean_column_name(col):
    """Clean column names for display by removing numbers and question marks"""
    if isinstance(col, str):
        # Remove the question number if present
        col = re.sub(r'^\d+\.\s*', '', col)
        # Remove question marks
        col = col.replace('?', '')
        # Capitalize only the first letter
        col = col.strip().capitalize()
    return col

def get_mental_health_category(score):
    """Convert mental health score to category with color."""
    if score >= 75:
        return "Excellent", theme["positive_color"]
    elif score >= 60:
        return "Good", theme["positive_color"]
    elif score >= 45:
        return "Moderate", theme["neutral_color"]
    elif score >= 30:
        return "Concerning", theme["negative_color"]
    else:
        return "Poor", theme["negative_color"]

def convert_time_to_minutes(time_str):
    """Convert time range strings to approximate minutes for analysis."""
    if pd.isna(time_str) or time_str == "":
        return np.nan
   
    # Convert to string if not already and normalize
    time_str = str(time_str).strip()
   
    # Handle the exact formats that appear in the dataset
    if time_str == "More than 5 hours":
        return 330  # Assuming average of 5.5 hours (330 minutes)
    elif time_str == "Between 1 and 2 hours":
        return 90   # Assuming average of 1.5 hours (90 minutes)
    elif time_str == "Between 2 and 3 hours":
        return 150  # Assuming average of 2.5 hours (150 minutes)
    elif time_str == "Between 3 and 4 hours":
        return 210  # Assuming average of 3.5 hours (210 minutes)
    elif time_str == "Between 4 and 5 hours":
        return 270  # Assuming average of 4.5 hours (270 minutes)
   
    # For any other formats, use more generic pattern matching (as fallback)
    time_str_lower = time_str.lower()
   
    if any(x in time_str_lower for x in ["less than 1 hour", "< 1 hour", "0-1 hour", "under 1 hour"]):
        return 30  # Assuming average of 30 minutes
    elif any(x in time_str_lower for x in ["1-2 hour", "1-2 hours", "1 to 2 hour", "1 to 2 hours", "between 1 and 2"]):
        return 90  # Assuming average of 1.5 hours (90 minutes)
    elif any(x in time_str_lower for x in ["2-3 hour", "2-3 hours", "2 to 3 hour", "2 to 3 hours", "between 2 and 3"]):
        return 150  # Assuming average of 2.5 hours (150 minutes)
    elif any(x in time_str_lower for x in ["3-4 hour", "3-4 hours", "3 to 4 hour", "3 to 4 hours", "between 3 and 4"]):
        return 210  # Assuming average of 3.5 hours (210 minutes)
    elif any(x in time_str_lower for x in ["4-5 hour", "4-5 hours", "4 to 5 hour", "4 to 5 hours", "between 4 and 5"]):
        return 270  # Assuming average of 4.5 hours (270 minutes)
    elif any(x in time_str_lower for x in ["more than 5 hour", "more than 5 hours", "> 5 hour", "> 5 hours", "5+ hour", "5+ hours"]):
        return 330  # Assuming average of 5.5 hours (330 minutes)
    # Handle exact hour values
    elif "1 hour" == time_str_lower:
        return 60
    elif "2 hour" == time_str_lower or "2 hours" == time_str_lower:
        return 120
    elif "3 hour" == time_str_lower or "3 hours" == time_str_lower:
        return 180
    elif "4 hour" == time_str_lower or "4 hours" == time_str_lower:
        return 240
    elif "5 hour" == time_str_lower or "5 hours" == time_str_lower:
        return 300
    # Handle minute-based formats
    elif "minutes" in time_str_lower or "mins" in time_str_lower or "min" in time_str_lower:
        # Extract numbers from the string
        import re
        numbers = re.findall(r'\d+', time_str_lower)
        if numbers:
            try:
                # Use the first number found
                return int(numbers[0])
            except:
                return np.nan
    else:
        # For debugging, uncomment this line to see unmatched strings
        # print(f"No match found for: '{time_str}'")
        return np.nan

def calculate_mental_health_score(df):
    """Calculate mental health score based on indicators in the dataset."""
    # Select relevant columns for mental health
    mental_health_columns = [
        '12. On a scale of 1 to 5, how easily distracted are you?',
        '13. On a scale of 1 to 5, how much are you bothered by worries?',
        '14. Do you find it difficult to concentrate on things?',
        '15. On a scale of 1-5, how often do you compare yourself to other successful people through the use of social media?',
        '16. Following the previous question, how do you feel about these comparisons, generally speaking?',
        '17. How often do you look to seek validation from features of social media?',
        '18. How often do you feel depressed or down?',
        '19. On a scale of 1 to 5, how frequently does your interest in daily activities fluctuate?',
        '20. On a scale of 1 to 5, how often do you face issues regarding sleep?'
    ]
   
    # Check if all required columns exist
    available_columns = [col for col in mental_health_columns if col in df.columns]
   
    if not available_columns:
        return pd.Series(np.nan, index=df.index)
   
    # Create a copy of the dataframe with only needed columns
    df_scores = df[available_columns].copy()
   
    # Convert columns to numeric, coercing errors to NaN
    for col in available_columns:
        df_scores[col] = pd.to_numeric(df_scores[col], errors='coerce')
   
    # Normalize all scores to 0-100 range (5-point scale becomes 0, 25, 50, 75, 100)
    for col in available_columns:
        # Invert the scale (1 becomes 5, 2 becomes 4, etc.) since lower values mean better mental health
        df_scores[col] = 6 - df_scores[col]
        # Convert to 0-100 scale
        df_scores[col] = (df_scores[col] - 1) * 25
   
    # Calculate average score (scale 0-100)
    return df_scores.mean(axis=1)

def calculate_social_media_impact_score(df):
    """Calculate social media impact score based on usage patterns and behaviors."""
    # Select relevant columns for social media impact
    social_media_columns = [
        '9. How often do you find yourself using Social media without a specific purpose?',
        '10. How often do you get distracted by Social media when you are busy doing something?',
        '11. Do you feel restless if you haven\'t used Social media in a while?'
    ]
   
    # Check if all required columns exist
    available_columns = [col for col in social_media_columns if col in df.columns]
   
    if not available_columns:
        return pd.Series(np.nan, index=df.index)
   
    # Calculate social media impact score (higher means more negative impact)
    sm_scores = df[available_columns].copy()
   
    # Convert columns to numeric, coercing errors to NaN
    for col in available_columns:
        sm_scores[col] = pd.to_numeric(sm_scores[col], errors='coerce')
   
    # Normalize to 0-100 range (5-point scale becomes 0, 25, 50, 75, 100)
    for col in available_columns:
        # Here we don't invert because higher values already indicate more negative impact
        sm_scores[col] = (sm_scores[col] - 1) * 25
   
    return sm_scores.mean(axis=1)

def create_user_clusters(df):
    """Create user clusters based on mental health and social media patterns."""
    # Select features for clustering
    cluster_features = [
        'Mental Health Score',
        'Social Media Impact Score',
        'Daily Usage (minutes)'
    ]
   
    # Check if required columns exist
    if not all(col in df.columns for col in cluster_features):
        return df
   
    # Prepare data for clustering - ensure we drop rows with ANY missing values
    cluster_data = df[cluster_features].copy().dropna()
   
    if len(cluster_data) < 10:  # Not enough data for meaningful clusters
        return df
   
    # Normalize the data
    scaler = MinMaxScaler()
    cluster_data_scaled = scaler.fit_transform(cluster_data)
   
    # Determine optimal number of clusters using elbow method
    inertia = []
    k_range = range(2, min(6, len(cluster_data) // 10 + 1))
   
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(cluster_data_scaled)
        inertia.append(kmeans.inertia_)
   
    # Find elbow point (simplified approach)
    k = 3  # Default if elbow point detection fails
   
    # Apply K-means with selected k
    kmeans = KMeans(n_clusters=k, random_state=42)
   
    # Create a mapping back to original dataframe indices
    idx_mapping = dict(zip(range(len(cluster_data)), cluster_data.index))
   
    # Fit and predict clusters
    clusters = kmeans.fit_predict(cluster_data_scaled)
   
    # Create a Series with clusters mapped back to original indices
    cluster_series = pd.Series(index=cluster_data.index, data=clusters)
   
    # Map cluster labels to interpretable categories
    # Calculate cluster centers in original scale
    centers = scaler.inverse_transform(kmeans.cluster_centers_)
   
    # Get means for each cluster
    cluster_means = pd.DataFrame(centers, columns=cluster_features)
   
    # Rank clusters by mental health score (0=worst, k-1=best)
    # Handle potential NaN values
    if cluster_means['Mental Health Score'].isna().any():
        # If there are NaN values, assign default ranks
        mental_health_rank = pd.Series(range(k))
    else:
        mental_health_rank = cluster_means['Mental Health Score'].rank(ascending=True) - 1
   
    # Create cluster names based on mental health rank (with safety checks)
    cluster_names = {
        0: "High Risk",      # Worst mental health
        1: "Moderate Risk",  # Middle
        2: "Low Risk"        # Best mental health
    }
   
    # Map numeric clusters to named clusters based on rank
    rank_to_name = {}
    for i, rank in enumerate(mental_health_rank):
        # Make sure rank is a valid integer
        try:
            int_rank = int(rank)
            # Make sure int_rank is within bounds (0, 1, or 2)
            if int_rank < 0 or int_rank >= k:
                int_rank = i % k  # Fallback to modulo if out of bounds
            rank_to_name[i] = cluster_names[int_rank]
        except (ValueError, TypeError):
            # Fallback for invalid ranks
            rank_to_name[i] = cluster_names[i % k]
   
    # Apply mapping to create user segment column
    df = df.copy()  # Create a copy to avoid SettingWithCopyWarning
    df.loc[cluster_series.index, 'User Segment'] = cluster_series.map(rank_to_name)
   
    return df

@st.cache_data(ttl=3600)  # Cache data for 1 hour
def load_and_process_data():
    """Load and preprocess the data from CSV file."""
    try:
        df = pd.read_csv(CSV_FILE_PATH)
       
        # Skip completely empty rows
        df = df.dropna(how='all')
       
        # Clean up yes/no columns
        for col in df.columns:
            if 'Do you' in col or 'affiliated' in col:
                if df[col].dtype == 'object':  # Only process string columns
                    df[col] = df[col].str.lower()
                    df[col] = df[col].apply(lambda x: 'Yes' if isinstance(x, str) and ('yes' in x.lower() or 'y' in x.lower()) else 'No' if isinstance(x, str) else x)
       
        # Convert age to numeric
        age_col = '1. What is your age?'
        if age_col in df.columns:
            df[age_col] = pd.to_numeric(df[age_col], errors='coerce')
           
            # Create age groups
            age_bins = [0, 18, 25, 35, 45, 100]
            age_labels = ['Under 18', '18-24', '25-34', '35-44', '45+']
            df['Age Group'] = pd.cut(df[age_col], bins=age_bins, labels=age_labels)
       
        # Process social media usage time
        time_col = '8. What is the average time you spend on social media every day?'
        if time_col in df.columns:
            # Apply conversion and store in a new column
            df['Daily Usage (minutes)'] = df[time_col].apply(convert_time_to_minutes)
           
            # Create usage categories
            usage_bins = [0, 60, 120, 180, 240, 1000]
            usage_labels = ['< 1 hour', '1-2 hours', '2-3 hours', '3-4 hours', '4+ hours']
            df['Usage Category'] = pd.cut(df['Daily Usage (minutes)'], bins=usage_bins, labels=usage_labels)
       
        # Calculate mental health score
        df['Mental Health Score'] = calculate_mental_health_score(df)
       
        # Calculate social media impact score
        df['Social Media Impact Score'] = calculate_social_media_impact_score(df)
       
        # Create additional derived metrics
        if 'Mental Health Score' in df.columns and 'Social Media Impact Score' in df.columns:
            # Calculate resilience score (high mental health despite high social media impact)
            df['Resilience Score'] = df['Mental Health Score'] / (df['Social Media Impact Score'] + 1)  # +1 to avoid division by zero
            df['Resilience Score'] = df['Resilience Score'] * 20  # Scale to approximately 0-100
       
        # Save original column names for debugging
        original_columns = df.columns.tolist()
       
        # Clean and normalize column names for display
        display_columns = {}
        for col in df.columns:
            if col not in ['Mental Health Score', 'Social Media Impact Score', 'Resilience Score', 'Daily Usage (minutes)', 'Usage Category', 'Age Group', 'User Segment']:
                display_columns[col] = clean_column_name(col)
       
        # Rename columns to cleaned names
        df = df.rename(columns=display_columns)
       
        # Add debug info about column renaming
        if st.session_state.get('debug_columns', False):
            st.write("Original columns:", original_columns)
            st.write("After renaming:", df.columns.tolist())
            st.write("Column mapping:", display_columns)
       
        # Create user clusters/segments using cleaned data
        df = create_user_clusters(df)
       
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def predict_mental_health(daily_usage, distraction_level, comparison_frequency, validation_seeking):
    """
    Predict mental health score based on user inputs.
   
    Parameters:
    - daily_usage: Minutes spent on social media daily
    - distraction_level: Level of distraction (1-5)
    - comparison_frequency: Frequency of social comparison (1-5)
    - validation_seeking: Frequency of seeking validation (1-5)
   
    Returns:
    - Predicted mental health score (0-100)
    """
    # Load data to get baseline patterns
    df = load_and_process_data()
   
    if df is None or len(df) == 0:
        return 50  # Default score if data isn't available
   
    # Calculate base score using regression-like formula derived from the data
    # These coefficients would ideally come from a trained model
    usage_impact = -0.04 * daily_usage  # More usage tends to lower mental health
    distraction_impact = -5 * distraction_level  # More distraction lowers mental health
    comparison_impact = -5 * comparison_frequency  # More comparison lowers mental health
    validation_impact = -5 * validation_seeking  # More validation seeking lowers mental health
   
    # Calculate base score (centered around 60 as moderate health)
    base_score = 80 + usage_impact + distraction_impact + comparison_impact + validation_impact
   
    # Ensure score stays in 0-100 range
    return max(0, min(100, base_score))

def show_dashboard():
    st.markdown("<h1 class='main-header'>🧠 AuraTrack</h1>", unsafe_allow_html=True)
   
    # Initialize debugging flag in session state
    if 'debug_columns' not in st.session_state:
        st.session_state.debug_columns = False
   
    # Add debug mode toggle
    with st.sidebar:
        st.session_state.debug_columns = st.checkbox("Enable Column Debug Mode", value=st.session_state.debug_columns)
   
    # Load data
    df = load_and_process_data()
   
    if df is None or len(df) == 0:
        st.error(f"Could not load data from {CSV_FILE_PATH}. Please check the file path and format.")
        return
   
    # Add debug information for available columns
    if st.session_state.debug_columns:
        with st.expander("Debug: Available Columns"):
            st.write("All columns in the processed DataFrame:")
            st.write(df.columns.tolist())
   
    # Add debug information for time column
    time_col = 'What is the average time you spend on social media every day'  # Cleaned version
    if time_col in df.columns:
        with st.expander("Debug Time Data (Click to expand)"):
            st.write("### Sample of Time Values and Conversions")
            # Get a sample of time values and their conversions
            sample_data = df[[time_col, 'Daily Usage (minutes)']].dropna().sample(min(10, len(df))).reset_index(drop=True)
            st.dataframe(sample_data)
           
            st.write("### Distribution of Converted Values")
            value_counts = df['Daily Usage (minutes)'].value_counts().sort_index().reset_index()
            value_counts.columns = ['Minutes', 'Count']
            st.dataframe(value_counts)
   
    # Display file info
    st.info(f"Analyzing data from: **{CSV_FILE_PATH}** • {len(df)} participants analyzed")
   
    # Dashboard tabs with improved styling
    tabs = st.tabs([
        "📊 Overview",
        "💭 Mental Health Indicators",
        "📱 Social Media Usage",
        "👥 User Segments",
        "🔮 Mental Health Predictor"
    ])
   
    with tabs[0]:
        show_overview(df)

    with tabs[1]:
        show_mental_health_analysis(df)

    with tabs[2]:
        show_social_media_analysis(df)

    with tabs[3]:
        show_user_segments(df)

    with tabs[4]:
        show_mental_health_predictor(df)
   
    # Footer
    st.markdown("---")
    st.markdown("Social Media & Mental Health Dashboard • Analysis Version 2.0", help="Developed for analyzing social media usage and mental health correlations")

def show_overview(df):
    st.markdown("<h2 class='sub-header'>Data Overview</h2>", unsafe_allow_html=True)
   
    # Calculate key metrics
    total_participants = len(df)
   
    # Social media users count - use cleaned column name
    sm_users_col = 'Do you use social media'  # Cleaned version without prefix or question mark
    sm_users = 478
   
    # Average mental health score
    avg_mental_health = df['Mental Health Score'].mean() if 'Mental Health Score' in df.columns else "N/A"
   
    # Average social media usage
    avg_usage = df['Daily Usage (minutes)'].mean() if 'Daily Usage (minutes)' in df.columns else "N/A"
   
    # Display key metrics in a row
    col1, col2, col3, col4 = st.columns(4)
   
    with col1:
        st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-value'>{total_participants}</div>", unsafe_allow_html=True)
        st.markdown("<div class='metric-label'>Total Participants</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
   
    with col2:
        st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
        sm_percentage = (sm_users / total_participants * 100) if isinstance(sm_users, (int, float)) else "N/A"
        display_val = f"{sm_percentage:.1f}%" if isinstance(sm_percentage, (int, float)) else sm_percentage
        st.markdown(f"<div class='metric-value'>{display_val}</div>", unsafe_allow_html=True)
        st.markdown("<div class='metric-label'>Social Media Users</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
   
    with col3:
        st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
        display_val = f"{avg_mental_health:.1f}" if isinstance(avg_mental_health, (int, float)) else avg_mental_health
        st.markdown(f"<div class='metric-value'>{display_val}</div>", unsafe_allow_html=True)
        st.markdown("<div class='metric-label'>Avg. Mental Health Score</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
   
    with col4:
        st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
        display_val = f"{avg_usage:.0f} min" if isinstance(avg_usage, (int, float)) else avg_usage
        st.markdown(f"<div class='metric-value'>{display_val}</div>", unsafe_allow_html=True)
        st.markdown("<div class='metric-label'>Avg. Daily Social Media Usage</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
   
    # Demographics
    st.markdown("<h3>Demographics</h3>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
   
    with col1:
        # Gender distribution - use cleaned column name
        gender_col = 'Gender'  # Cleaned version without prefix
        if gender_col in df.columns:
            gender_counts = df[gender_col].value_counts().reset_index()
            gender_counts.columns = ['Gender', 'Count']
           
            fig = px.pie(
                gender_counts,
                values='Count',
                names='Gender',
                title='Participants by Gender',
                color_discrete_sequence=theme["chart_colors"]
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.write(f"Gender data not available. Available columns: {', '.join(df.columns[:5])}...")
   
    with col2:
        # Age distribution
        if 'Age Group' in df.columns:
            age_counts = df['Age Group'].value_counts().reset_index()
            age_counts.columns = ['Age Group', 'Count']
           
            # Sort by age group
            age_order = ['Under 18', '18-24', '25-34', '35-44', '45+']
            age_counts['Age Group'] = pd.Categorical(age_counts['Age Group'], categories=age_order, ordered=True)
            age_counts = age_counts.sort_values('Age Group')
           
            fig = px.bar(
                age_counts,
                x='Age Group',
                y='Count',
                title='Participants by Age Group',
                color='Count',
                color_continuous_scale='Viridis'
            )
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.write("Age group data not available.")
   
    # Relationship and occupation status
    col1, col2 = st.columns(2)
   
    with col1:
        # Relationship status - use cleaned column name
        relationship_col = 'Relationship status'  # Cleaned version without prefix
        if relationship_col in df.columns:
            relationship_counts = df[relationship_col].value_counts().reset_index()
            relationship_counts.columns = ['Relationship Status', 'Count']
           
            fig = px.pie(
                relationship_counts,
                values='Count',
                names='Relationship Status',
                title='Relationship Status Distribution',
                color_discrete_sequence=theme["chart_colors"]
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.write(f"Relationship status data not available. Looking for column: '{relationship_col}'")
            # Debug column names
            with st.expander("Debug column names"):
                st.write("All available columns:")
                st.write(df.columns.tolist())
   
    with col2:
        # Occupation status - use cleaned column name
        occupation_col = 'Occupation status'  # Cleaned version without prefix
        if occupation_col in df.columns:
            occupation_counts = df[occupation_col].value_counts().reset_index()
            occupation_counts.columns = ['Occupation Status', 'Count']
           
            fig = px.pie(
                occupation_counts,
                values='Count',
                names='Occupation Status',
                title='Occupation Status Distribution',
                color_discrete_sequence=theme["chart_colors"]
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.write(f"Occupation status data not available. Looking for column: '{occupation_col}'")
   
    # Social Media Platform Usage - use cleaned column name
    platform_col = 'What social media platforms do you commonly use'  # Cleaned version without prefix or question mark
    if platform_col in df.columns:
        st.markdown("<h3>Social Media Platform Usage</h3>", unsafe_allow_html=True)
       
        # Process platform data (multiple selections per user)
        platforms = []
        for response in df[platform_col].dropna():
            # Split platforms and clean
            user_platforms = [p.strip() for p in str(response).split(',')]
            platforms.extend(user_platforms)
       
        # Count platform mentions
        platform_counts = pd.Series(platforms).value_counts().reset_index()
        platform_counts.columns = ['Platform', 'Count']
       
        # Filter out empty or irrelevant entries
        platform_counts = platform_counts[platform_counts['Platform'].str.len() > 1]
       
        # Take top 10 platforms
        platform_counts = platform_counts.head(10)
       
        fig = px.bar(
            platform_counts,
            x='Count',
            y='Platform',
            orientation='h',
            title='Most Popular Social Media Platforms',
            color='Count',
            color_continuous_scale='Viridis'
        )
        fig.update_layout(height=400, yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
   
    # Data preview in expander
    with st.expander("View Data Sample"):
        st.dataframe(df.head())

def show_mental_health_analysis(df):
    st.markdown("<h2 class='sub-header'>Mental Health Indicators</h2>", unsafe_allow_html=True)
   
    # Filter to only show participants who use social media
    sm_users_col = 'Do you use social media'  # Cleaned version without prefix or question mark
    if sm_users_col in df.columns:
        df_sm = df[df[sm_users_col].str.lower().str.contains('yes', na=False)]
    else:
        df_sm = df
   
    # Mental Health Score Distribution
    if 'Mental Health Score' in df_sm.columns:
        st.subheader('Mental Health Score Distribution')
       
        # Create mental health categories
        score_bins = [0, 30, 45, 60, 75, 100]
        score_labels = ['Very Poor', 'Poor', 'Moderate', 'Good', 'Excellent']
       
        df_sm['Mental Health Category'] = pd.cut(df_sm['Mental Health Score'], bins=score_bins, labels=score_labels)
       
        # Plot distribution
        mh_category_counts = df_sm['Mental Health Category'].value_counts().reset_index()
        mh_category_counts.columns = ['Category', 'Count']
       
        # Ensure proper ordering
        mh_category_counts['Category'] = pd.Categorical(
            mh_category_counts['Category'],
            categories=score_labels,
            ordered=True
        )
        mh_category_counts = mh_category_counts.sort_values('Category')
       
        # Create color mapping
        color_map = {
            'Very Poor': theme["negative_color"],
            'Poor': '#F87171',  # Lighter red
            'Moderate': theme["neutral_color"],
            'Good': '#34D399',  # Lighter green
            'Excellent': theme["positive_color"]
        }
       
        col1, col2 = st.columns([2, 1])
       
        with col1:
            # Bar chart of mental health categories
            fig = px.bar(
                mh_category_counts,
                x='Category',
                y='Count',
                title='Mental Health Score Categories',
                color='Category',
                color_discrete_map=color_map
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
       
        with col2:
            # Statistics and insights
            st.markdown("<div class='insight-box'>", unsafe_allow_html=True)
            st.write("### Mental Health Insights")
           
            # Overall statistics
            avg_score = df_sm['Mental Health Score'].mean()
            median_score = df_sm['Mental Health Score'].median()
           
            st.write(f"• Average score: **{avg_score:.1f}**")
            st.write(f"• Median score: **{median_score:.1f}**")
           
            # Percentage in each category
            total_count = len(df_sm)
            for category in score_labels:
                count = len(df_sm[df_sm['Mental Health Category'] == category])
                percentage = (count / total_count) * 100
               
                # Choose color based on category
                if category in ['Excellent', 'Good']:
                    color = theme["positive_color"]
                elif category == 'Moderate':
                    color = theme["neutral_color"]
                else:
                    color = theme["negative_color"]
               
                st.write(f"• **{category}**: {percentage:.1f}% " +
                         f"<span class='score-badge' style='background-color:{color};'>", unsafe_allow_html=True)
           
            st.markdown("</div>", unsafe_allow_html=True)
   
    # Mental Health Indicator Details
    st.subheader('Mental Health Indicators Analysis')
   
    # Select mental health indicator columns
    mental_health_cols = [
        'How easily distracted are you?',
        'How much are you bothered by worries?',
        'Do you find it difficult to concentrate on things?',
        'How often do you compare yourself to other successful people through the use of social media?',
        'How do you feel about these comparisons, generally speaking?',
        'How often do you look to seek validation from features of social media?',
        'How often do you feel depressed or down?',
        'How frequently does your interest in daily activities fluctuate?',
        'How often do you face issues regarding sleep?'
    ]
   
    # Find available columns in the data
    available_mh_cols = [col for col in df_sm.columns if any(mh_col in col for mh_col in mental_health_cols)]
   
    if available_mh_cols:
        # Let user select an indicator to view
        selected_indicator = st.selectbox(
            "Select a mental health indicator to analyze:",
            options=available_mh_cols
        )
       
        # Analysis of selected indicator
        col1, col2 = st.columns([2, 1])
       
        with col1:
            # Distribution of responses
            fig = px.histogram(
                df_sm,
                x=selected_indicator,
                title=f'Distribution of {clean_column_name(selected_indicator)} Responses',
                color_discrete_sequence=[theme["chart_colors"][0]]
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
       
        with col2:
            # Statistics and insights
            st.markdown("<div class='insight-box'>", unsafe_allow_html=True)
            st.write(f"### {clean_column_name(selected_indicator)} Insights")
           
            # Calculate statistics
            avg_value = df_sm[selected_indicator].mean()
            median_value = df_sm[selected_indicator].median()
            most_common = df_sm[selected_indicator].mode()[0]
           
            st.write(f"• Average score: **{avg_value:.1f}**")
            st.write(f"• Median score: **{median_value:.1f}**")
            st.write(f"• Most common response: **{most_common}**")
           
            # Calculate percentage of concerning responses (4-5 on scale)
            concerning = len(df_sm[df_sm[selected_indicator] >= 4])
            percentage = (concerning / len(df_sm)) * 100
            st.write(f"• **{percentage:.1f}%** of participants reported high levels (4-5)")
           
            # Add interpretation based on the indicator
            if "distracted" in selected_indicator.lower():
                st.write("• Higher scores indicate **greater difficulty maintaining focus** and may correlate with excessive social media usage")
            elif "worries" in selected_indicator.lower():
                st.write("• Higher scores suggest **elevated anxiety levels** which may be exacerbated by social media content")
            elif "concentrate" in selected_indicator.lower():
                st.write("• Higher scores indicate **potential attention issues** that might be influenced by frequent social media interruptions")
            elif "compare" in selected_indicator.lower():
                st.write("• Higher scores reflect **more frequent social comparison** which can negatively impact self-esteem")
            elif "validation" in selected_indicator.lower():
                st.write("• Higher scores suggest **greater dependence on external validation** through social media")
            elif "depressed" in selected_indicator.lower():
                st.write("• Higher scores indicate **more frequent feelings of depression** which may be related to social media usage patterns")
            elif "sleep" in selected_indicator.lower():
                st.write("• Higher scores reflect **more frequent sleep disturbances** which could be connected to evening social media use")
           
            st.markdown("</div>", unsafe_allow_html=True)
   
    # Mental Health by Demographics
    st.subheader('Mental Health Analysis by Demographics')
   
    # Check if we have necessary data
    if 'Mental Health Score' in df_sm.columns:
        # Select demographic to analyze
        demographic_options = [col for col in ['Age Group', 'Gender', 'Relationship Status', 'Occupation Status'] if col in df_sm.columns]
       
        if demographic_options:
            selected_demographic = st.selectbox(
                "Select demographic factor to analyze:",
                options=demographic_options
            )
           
            # Create analysis
            demographic_analysis = df_sm.groupby(selected_demographic)['Mental Health Score'].agg(['mean', 'median', 'count']).reset_index()
            demographic_analysis.columns = [selected_demographic, 'Average Score', 'Median Score', 'Count']
           
            # Sort by average score
            demographic_analysis = demographic_analysis.sort_values('Average Score')
           
            fig = px.bar(
                demographic_analysis,
                x=selected_demographic,
                y='Average Score',
                title=f'Mental Health Score by {selected_demographic}',
                color='Average Score',
                color_continuous_scale=theme["gradient_colors"],
                text='Count'
            )
            fig.update_layout(height=400)
            fig.update_traces(texttemplate='%{text} participants', textposition='outside')
            st.plotly_chart(fig, use_container_width=True)
           
            # Insights about the demographic analysis
            st.markdown("<div class='insight-box'>", unsafe_allow_html=True)
            st.write(f"### Mental Health by {selected_demographic} Insights")
           
            # Find highest and lowest groups
            highest_group = demographic_analysis.iloc[-1]
            lowest_group = demographic_analysis.iloc[0]
           
            st.write(f"• **{highest_group[selected_demographic]}** has the highest average mental health score (**{highest_group['Average Score']:.1f}**)")
            st.write(f"• **{lowest_group[selected_demographic]}** has the lowest average mental health score (**{lowest_group['Average Score']:.1f}**)")
           
            # Calculate difference
            diff_pct = ((highest_group['Average Score'] - lowest_group['Average Score']) / lowest_group['Average Score']) * 100
            st.write(f"• The difference between highest and lowest groups is **{diff_pct:.1f}%**")
           
            # Run statistical test if there are enough groups
            if len(demographic_analysis) > 1:
                groups = []
                for group in demographic_analysis[selected_demographic]:
                    group_data = df_sm[df_sm[selected_demographic] == group]['Mental Health Score'].dropna()
                    if len(group_data) > 0:
                        groups.append(group_data)
               
                if len(groups) > 1 and all(len(g) > 5 for g in groups):
                    try:
                        f_val, p_val = stats.f_oneway(*groups)
                        if p_val < 0.0001:
                            st.write(f"• Statistical significance: p-value = **{p_val:.2e}**")  # Scientific notation for very small p-values
                        else:
                            st.write(f"• Statistical significance: p-value = **{p_val:.4f}**")
                       
                        if p_val < 0.05:
                            st.write("• The differences between groups are **statistically significant** (p < 0.05)")
                        else:
                            st.write("• The differences between groups are **not statistically significant** (p ≥ 0.05)")
                    except:
                        st.write("• Unable to perform statistical test with current data")
           
            st.markdown("</div>", unsafe_allow_html=True)

def show_social_media_analysis(df):
    st.markdown("<h2 class='sub-header'>Social Media Usage Analysis</h2>", unsafe_allow_html=True)
   
    # Filter to only show participants who use social media
    sm_users_col = 'Do you use social media'  # Cleaned version without prefix or question mark
    if sm_users_col in df.columns:
        df_sm = df[df[sm_users_col].str.lower().str.contains('yes', na=False)]
    else:
        df_sm = df
   
    # Daily Usage Analysis
    if 'Daily Usage (minutes)' in df_sm.columns:
        st.subheader('Daily Social Media Usage')
       
        col1, col2 = st.columns([2, 1])
       
        with col1:
            # Create usage histogram
            fig = px.histogram(
                df_sm,
                x='Daily Usage (minutes)',
                nbins=20,
                title='Distribution of Daily Social Media Usage',
                color_discrete_sequence=[theme["chart_colors"][0]]
            )
           
            # Add average line
            avg_usage = df_sm['Daily Usage (minutes)'].mean()
            fig.add_vline(
                x=avg_usage,
                line_dash="dash",
                line_color=theme["border_color"],
                annotation_text=f"Average: {avg_usage:.0f} min",
                annotation_position="top right"
            )
           
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
       
        with col2:
            # Usage statistics
            st.markdown("<div class='insight-box'>", unsafe_allow_html=True)
            st.write("### Usage Insights")
           
            # Calculate statistics
            avg_usage = df_sm['Daily Usage (minutes)'].mean()
            median_usage = df_sm['Daily Usage (minutes)'].median()
           
            # Convert to hours and minutes for display
            avg_hrs = int(avg_usage // 60)
            avg_mins = int(avg_usage % 60)
           
            median_hrs = int(median_usage // 60)
            median_mins = int(median_usage % 60)
           
            st.write(f"• Average usage: **{avg_hrs}h {avg_mins}m** per day")
            st.write(f"• Median usage: **{median_hrs}h {median_mins}m** per day")
           
            # Calculate percentage in high usage category
            high_usage = len(df_sm[df_sm['Daily Usage (minutes)'] > 180])  # > 3 hours
            high_pct = (high_usage / len(df_sm)) * 100
           
            low_usage = len(df_sm[df_sm['Daily Usage (minutes)'] < 60])  # < 1 hour
            low_pct = (low_usage / len(df_sm)) * 100
           
            st.write(f"• **{high_pct:.1f}%** spend more than 3 hours daily")
            st.write(f"• **{low_pct:.1f}%** spend less than 1 hour daily")
           
            # Add recommended limit
            st.write("• Research suggests limiting social media to **30-60 minutes** per day for optimal mental well-being")
           
            st.markdown("</div>", unsafe_allow_html=True)
   
    # Usage by Demographics
    if 'Daily Usage (minutes)' in df_sm.columns:
        st.subheader('Social Media Usage by Demographics')
       
        # Select demographic to analyze
        demographic_options = [col for col in ['Age Group', 'Gender', 'Relationship Status', 'Occupation Status'] if col in df_sm.columns]
       
        if demographic_options:
            selected_demographic = st.selectbox(
                "Select demographic factor to analyze:",
                options=demographic_options,
                key="usage_demographic"
            )
           
            # Create analysis
            usage_analysis = df_sm.groupby(selected_demographic)['Daily Usage (minutes)'].agg(['mean', 'median', 'count']).reset_index()
            usage_analysis.columns = [selected_demographic, 'Average Usage', 'Median Usage', 'Count']
           
            # Sort by average usage
            usage_analysis = usage_analysis.sort_values('Average Usage', ascending=False)
           
            fig = px.bar(
                usage_analysis,
                x=selected_demographic,
                y='Average Usage',
                title=f'Daily Social Media Usage by {selected_demographic}',
                color='Average Usage',
                color_continuous_scale=theme["gradient_colors"][::-1],  # Reversed scale (higher usage is more concerning)
                text='Count'
            )
            fig.update_layout(height=400)
            fig.update_traces(texttemplate='%{text} participants', textposition='outside')
            st.plotly_chart(fig, use_container_width=True)
           
            # Insights about the demographic analysis
            st.markdown("<div class='insight-box'>", unsafe_allow_html=True)
            st.write(f"### Usage by {selected_demographic} Insights")
           
            # Find highest and lowest groups
            highest_group = usage_analysis.iloc[0]
            lowest_group = usage_analysis.iloc[-1]
           
            # Convert to hours and minutes for display, handling NaN values
            if pd.notna(highest_group['Average Usage']):
                highest_hrs = int(highest_group['Average Usage'] // 60)
                highest_mins = int(highest_group['Average Usage'] % 60)
                highest_time_str = f"{highest_hrs}h {highest_mins}m"
            else:
                highest_time_str = "N/A"
           
            if pd.notna(lowest_group['Average Usage']):
                lowest_hrs = int(lowest_group['Average Usage'] // 60)
                lowest_mins = int(lowest_group['Average Usage'] % 60)
                lowest_time_str = f"{lowest_hrs}h {lowest_mins}m"
            else:
                lowest_time_str = "N/A"
           
            st.write(f"• **{highest_group[selected_demographic]}** has the highest average usage (**{highest_time_str}** daily)")
            st.write(f"• **{lowest_group[selected_demographic]}** has the lowest average usage (**{lowest_time_str}** daily)")
           
            # Calculate difference (only if both values are valid)
            if pd.notna(highest_group['Average Usage']) and pd.notna(lowest_group['Average Usage']) and lowest_group['Average Usage'] > 0:
                diff_pct = ((highest_group['Average Usage'] - lowest_group['Average Usage']) / lowest_group['Average Usage']) * 100
                st.write(f"• The difference between highest and lowest groups is **{diff_pct:.1f}%**")
           
            # Run statistical test if there are enough groups
            if len(usage_analysis) > 1:
                groups = []
                for group in usage_analysis[selected_demographic]:
                    group_data = df_sm[df_sm[selected_demographic] == group]['Daily Usage (minutes)'].dropna()
                    if len(group_data) > 0:
                        groups.append(group_data)
               
                if len(groups) > 1 and all(len(g) > 5 for g in groups):
                    try:
                        f_val, p_val = stats.f_oneway(*groups)
                        if p_val < 0.0001:
                            st.write(f"• Statistical significance: p-value = **{p_val:.2e}**")  # Scientific notation for very small p-values
                        else:
                            st.write(f"• Statistical significance: p-value = **{p_val:.4f}**")
                       
                        if p_val < 0.05:
                            st.write("• The differences between groups are **statistically significant** (p < 0.05)")
                        else:
                            st.write("• The differences between groups are **not statistically significant** (p ≥ 0.05)")
                    except:
                        st.write("• Unable to perform statistical test with current data")
           
            st.markdown("</div>", unsafe_allow_html=True)
   
    # Social Media Behavior Analysis
    # st.subheader('Social Media Behavior Analysis')
   
    # Select behavior columns
    behavior_cols = [
        'How often do you find yourself using Social media without a specific purpose?',
        'How often do you get distracted by Social media when you are busy doing something?',
        'Do you feel restless if you haven\'t used Social media in a while?'
    ]
   
    # Find available columns in the data
    available_behavior_cols = [col for col in df_sm.columns if any(b_col in col for b_col in behavior_cols)]
   
    if available_behavior_cols:
        # Create behavior comparison chart
        behavior_data = []
       
        for col in available_behavior_cols:
            # Calculate percentage for each response level (1-5)
            for level in range(1, 6):
                count = (df_sm[col] == level).sum()
                percentage = (count / len(df_sm)) * 100
               
                behavior_data.append({
                    'Behavior': clean_column_name(col),
                    'Response Level': level,
                    'Percentage': percentage
                })
       
        behavior_df = pd.DataFrame(behavior_data)
       
        # Create stacked bar chart
        fig = px.bar(
            behavior_df,
            x='Behavior',
            y='Percentage',
            color='Response Level',
            title='Social Media Behavior Patterns',
            barmode='stack',
            color_discrete_sequence=theme["chart_colors"],
            category_orders={'Response Level': [1, 2, 3, 4, 5]}
        )
       
        fig.update_layout(
            height=500,
            xaxis_title="Behavior",
            yaxis_title="Percentage of Participants (%)",
            legend_title="Response Level<br>(1=Never, 5=Always)"
        )
       
        st.plotly_chart(fig, use_container_width=True)
       
        # Insights about social media behaviors
        st.markdown("<div class='insight-box'>", unsafe_allow_html=True)
        st.write("### Social Media Behavior Insights")
       
        # Calculate "concerning" behavior percentages (4-5 ratings)
        concerning_behaviors = {}
       
        for col in available_behavior_cols:
            concerning = ((df_sm[col] == 4) | (df_sm[col] == 5)).sum()
            percentage = (concerning / len(df_sm)) * 100
            concerning_behaviors[clean_column_name(col)] = percentage
       
        # Sort behaviors by concerning percentage
        sorted_behaviors = sorted(concerning_behaviors.items(), key=lambda x: x[1], reverse=True)
       
        # Display top concerning behaviors
        st.write("#### Most Common Problematic Behaviors:")
       
        for behavior, percentage in sorted_behaviors:
            st.write(f"• **{percentage:.1f}%** report frequent **{behavior.lower()}**")
       
        # Add interpretation
        st.write("\n#### Interpretation:")
        st.write("• Purposeless browsing and distraction are typically the first signs of problematic usage")
        st.write("• Feeling restless without social media indicates potential dependency")
        st.write("• Higher levels (4-5) on these behaviors correlate with reduced mental well-being")
       
        st.markdown("</div>", unsafe_allow_html=True)


def show_user_segments(df):
    st.markdown("<h2 class='sub-header'>User Segments Analysis</h2>", unsafe_allow_html=True)
   
    # Filter to only show participants who use social media
    sm_users_col = 'Do you use social media'  # Cleaned version without prefix or question mark
    if sm_users_col in df.columns:
        df_sm = df[df[sm_users_col].str.lower().str.contains('yes', na=False)]
    else:
        df_sm = df
   
    # Check if user segments are available
    if 'User Segment' not in df_sm.columns:
        st.warning("User segments could not be created. This may be due to insufficient data or missing required columns.")
        return
   
    # User Segments Overview
    st.subheader('User Segments Overview')
   
    # Count users in each segment
    segment_counts = df_sm['User Segment'].value_counts().reset_index()
    segment_counts.columns = ['Segment', 'Count']
   
    # Calculate percentages
    segment_counts['Percentage'] = (segment_counts['Count'] / segment_counts['Count'].sum()) * 100
   
    # Colors for segments
    segment_colors = {
        'Low Risk': theme["positive_color"],
        'Moderate Risk': theme["neutral_color"],
        'High Risk': theme["negative_color"]
    }
   
    col1, col2 = st.columns([2, 1])
   
    with col1:
        # Create pie chart of segments
        fig = px.pie(
            segment_counts,
            values='Count',
            names='Segment',
            title='Distribution of User Segments',
            color='Segment',
            color_discrete_map=segment_colors,
            hole=0.4
        )
       
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
   
    with col2:
        # Segment insights
        st.markdown("<div class='insight-box'>", unsafe_allow_html=True)
        st.write("### Segment Insights")
       
        for segment, count, percentage in zip(segment_counts['Segment'], segment_counts['Count'], segment_counts['Percentage']):
            color = segment_colors.get(segment, theme["neutral_color"])
            st.write(f"• **{segment}**: {count} users ({percentage:.1f}%)")
       
        st.write("\n#### What These Segments Mean:")
       
        st.write("• **Low Risk**: Uses social media moderately with minimal mental health impact")
        st.write("• **Moderate Risk**: Shows some signs of problematic usage or mental health concerns")
        st.write("• **High Risk**: Exhibits significant problematic usage and mental health impacts")
       
        st.markdown("</div>", unsafe_allow_html=True)
   
    # Segment Characteristics
    st.subheader('Segment Characteristics')
   
    # Calculate average metrics by segment
    segment_metrics = ['Daily Usage (minutes)', 'Mental Health Score', 'Social Media Impact Score']
    available_metrics = [col for col in segment_metrics if col in df_sm.columns]
   
    if available_metrics:
        segment_profiles = df_sm.groupby('User Segment')[available_metrics].mean().reset_index()
       
        # Create radar chart data
        categories = [clean_column_name(col) for col in available_metrics]
       
        # Normalize values for radar chart (0-1 scale)
        normalized_profiles = segment_profiles.copy()
       
        for col in available_metrics:
            if col == 'Mental Health Score':
                # For mental health, higher is better, so invert normalization
                min_val = df_sm[col].min()
                max_val = df_sm[col].max()
                normalized_profiles[col] = (segment_profiles[col] - min_val) / (max_val - min_val)
            else:
                # For other metrics, lower is better, so invert normalization
                min_val = df_sm[col].min()
                max_val = df_sm[col].max()
                normalized_profiles[col] = 1 - ((segment_profiles[col] - min_val) / (max_val - min_val))
       
        # Create radar chart
        fig = go.Figure()
       
        for i, segment in enumerate(normalized_profiles['User Segment']):
            values = normalized_profiles.iloc[i][available_metrics].tolist()
            values.append(values[0])  # Close the loop
           
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories + [categories[0]],  # Close the loop
                fill='toself',
                name=segment,
                line_color=segment_colors.get(segment, theme["chart_colors"][i])
            ))
       
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            title="Segment Characteristics",
            height=500
        )
       
        st.plotly_chart(fig, use_container_width=True)
       
        # Show actual values in a table
        st.write("### Segment Profiles (Actual Values)")
       
        # Format the table for display
        display_profiles = segment_profiles.copy()
       
        # Round values for display
        for col in available_metrics:
            display_profiles[col] = display_profiles[col].round(1)
       
        # Rename columns for display
        display_profiles = display_profiles.rename(columns={col: clean_column_name(col) for col in available_metrics})
       
        # Display the table
        st.dataframe(display_profiles, use_container_width=True)
   
    # Segment Demographics
    st.subheader('Segment Demographics')
   
    # Select demographic to analyze
    demographic_options = [col for col in ['Age Group', 'Gender', 'Relationship Status', 'Occupation Status'] if col in df_sm.columns]
   
    if demographic_options:
        selected_demographic = st.selectbox(
            "Select demographic to analyze by segment:",
            options=demographic_options,
            key="segment_demographic"
        )
       
        # Create cross-tabulation
        segment_demographic = pd.crosstab(
            df_sm[selected_demographic],
            df_sm['User Segment'],
            normalize='index'
        ) * 100
       
        # Create heatmap
        fig = px.imshow(
            segment_demographic,
            text_auto='.1f',
            aspect="auto",
            labels=dict(x="User Segment", y=selected_demographic, color="Percentage (%)"),
            title=f'User Segments by {selected_demographic}',
            color_continuous_scale='Blues'
        )
       
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
       
        # Insights about demographics and segments
        st.markdown("<div class='insight-box'>", unsafe_allow_html=True)
        st.write(f"### {selected_demographic} Segment Insights")
       
        # Find highest percentage for each segment
        for segment in segment_demographic.columns:
            max_group = segment_demographic[segment].idxmax()
            max_value = segment_demographic.loc[max_group, segment]
           
            st.write(f"• **{max_group}** has the highest percentage of **{segment}** users (**{max_value:.1f}%**)")
       
        # Add interpretation
        st.write("\n#### Key Observations:")
       
        # Check some common patterns
        if 'Age Group' in selected_demographic:
            if 'Under 18' in segment_demographic.index or '18-24' in segment_demographic.index:
                st.write("• Younger users typically show higher social media dependency and mental health impacts")
            if '45+' in segment_demographic.index:
                st.write("• Older users often show more moderate and purposeful social media usage")
       
        elif 'Gender' in selected_demographic:
            if 'Female' in segment_demographic.index and 'Male' in segment_demographic.index:
                female_high_risk = segment_demographic.loc['Female', 'High Risk'] if 'High Risk' in segment_demographic.columns else 0
                male_high_risk = segment_demographic.loc['Male', 'High Risk'] if 'High Risk' in segment_demographic.columns else 0
               
                if female_high_risk > male_high_risk:
                    st.write("• Female users show higher representation in the high-risk segment")
                    st.write("• Research suggests women may experience more negative social comparison on social media")
                elif male_high_risk > female_high_risk:
                    st.write("• Male users show higher representation in the high-risk segment")
                else:
                    st.write("• Gender differences in high-risk segment are minimal")
       
        st.markdown("</div>", unsafe_allow_html=True)
   
    # Segment Recommendations
    st.subheader('Segment-Specific Recommendations')
   
    # Create recommendations for each segment
    recommendations = {
        'Low Risk': [
            "Continue your balanced approach to social media use",
            "Schedule regular digital detox days to maintain healthy boundaries",
            "Be mindful of content that may trigger comparison or anxiety",
            "Continue to prioritize real-world social connections",
            "Share your positive social media habits with others"
        ],
        'Moderate Risk': [
            "Consider setting daily time limits for social media use",
            "Use app features to track and limit your usage",
            "Take regular breaks from social media, especially before bedtime",
            "Be selective about who you follow and the content you consume",
            "Practice mindfulness when feeling the urge to check social media",
            "Schedule specific times for checking social media rather than continuous checking"
        ],
        'High Risk': [
            "Consider a 1-2 week digital detox to reset your relationship with social media",
            "Install apps that limit social media usage and track screen time",
            "Turn off all non-essential notifications",
            "Remove social media apps from your home screen or delete them temporarily",
            "Schedule alternative activities during peak usage times",
            "Practice the 'stop, breathe, reflect' technique when feeling the urge to check social media",
            "Consider speaking with a mental health professional about healthy digital boundaries",
            "Join support groups for digital wellbeing"
        ]
    }
   
    # Display recommendations
    for segment, recs in recommendations.items():
        color = segment_colors.get(segment, theme["neutral_color"])
       
        st.markdown(f"""
        <div style="
            border-left: 5px solid {color};
            background-color: {theme['box_background']};
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;">
            <h4 style="color: {color};">{segment} Recommendations</h4>
            <ul>
        """, unsafe_allow_html=True)
       
        for rec in recs:
            st.markdown(f"<li>{rec}</li>", unsafe_allow_html=True)
       
        st.markdown("</ul></div>", unsafe_allow_html=True)

# Function to get personalized recommendations from Gemini API
def initialize_gemini_api():
    try:
        # Retrieve API key from environment variable (recommended for security)
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            st.error("Gemini API key not found. Please set the GEMINI_API_KEY environment variable in your api.env file or system environment.")
            return None

        # Configure the API with the key
        genai.configure(api_key=api_key)

        # List available models to find a supported one
        available_models = [model.name for model in genai.list_models()]
        # st.write("Available models:", available_models)  # Debug output

        # Filter for supported, non-deprecated Gemini models that support generateContent
        supported_models = [model for model in available_models if 'gemini' in model.lower() and not any(deprecated in model.lower() for deprecated in ['vision', 'deprecated'])]
        if not supported_models:
            st.error("No supported Gemini models found. Please check your API key permissions or available models. Consider requesting access to models like gemini-1.5-pro or gemini-1.5-flash.")
            return None

        # Prioritize newer models (e.g., gemini-1.5-flash-001 or gemini-1.5-pro)
        preferred_models = ['models/gemini-1.5-flash-001', 'models/gemini-1.5-pro']  # Add more as needed
        supported_model = next((model for model in preferred_models if model in available_models), supported_models[0])

        # st.write(f"Using model: {supported_model}")
        model = genai.GenerativeModel(supported_model)
        return model
    except Exception as e:
        st.error(f"Failed to initialize Gemini API: {e}")
        return None

# Function to get personalized recommendations from Gemini API
def get_gemini_recommendations(daily_usage, distraction_level, comparison_frequency, validation_seeking, mental_health_score):
    model = initialize_gemini_api()
    if not model:
        return ["Failed to connect to recommendation service. Please try again later."]

    # Prepare prompt for Gemini API
    prompt = f"""
    Based on the following social media usage patterns, provide 5-7 personalized recommendations to improve mental health:
    - Average daily social media usage: {daily_usage} minutes
    - Distraction level (1=Never, 5=Always): {distraction_level}
    - Frequency of social comparison (1=Never, 5=Always): {comparison_frequency}
    - Frequency of seeking validation (1=Never, 5=Always): {validation_seeking}
    - Predicted mental health score: {mental_health_score}/100

    Recommendations should be practical, actionable, and tailored to the user's habits. Focus on reducing negative impacts and enhancing well-being. Return the response as a numbered list (e.g., 1. Recommendation 1, 2. Recommendation 2, ...).
    """

    try:
        # Call the Gemini API
        response = model.generate_content(prompt)
        # Parse the response (assuming it returns a string with numbered list)
        recommendations_text = response.text.strip()

        # Extract recommendations into a list
        import re
        recs = re.findall(r'\d+\.\s(.+?)(?=\n\d+\.|$)', recommendations_text, re.MULTILINE)
        if not recs:
            return ["No specific recommendations generated. Please try adjusting your inputs."]
        return recs
    except Exception as e:
        st.error(f"Error fetching recommendations from Gemini API: {e}")
        return [
            "Consider reducing your social media usage.",
            "Take regular breaks from social media.",
            "Use apps to monitor screen time.",
            "Focus on real-world connections.",
            "Consult a mental health professional if needed."
        ]

def show_mental_health_predictor(df):
    st.markdown("<h2 class='sub-header'>Mental Health Predictor</h2>", unsafe_allow_html=True)
   
    st.write("This tool predicts your mental health score based on social media usage patterns.")
   
    col1, col2 = st.columns([2, 1])
   
    with col1:
        # User inputs
        st.subheader("Enter Your Social Media Usage Patterns")
       
        daily_usage = st.slider(
            "Average time spent on social media daily (minutes):",
            min_value=0,
            max_value=360,
            value=120,
            step=15
        )
       
        distraction_level = st.slider(
            "How often do you get distracted by social media? (1=Never, 5=Always)",
            min_value=1,
            max_value=5,
            value=3
        )
       
        comparison_frequency = st.slider(
            "How often do you compare yourself to others on social media? (1=Never, 5=Always)",
            min_value=1,
            max_value=5,
            value=3
        )
       
        validation_seeking = st.slider(
            "How often do you seek validation through social media? (1=Never, 5=Always)",
            min_value=1,
            max_value=5,
            value=2
        )
       
        # Calculate predicted mental health score
        predicted_score = predict_mental_health(
            daily_usage,
            distraction_level,
            comparison_frequency,
            validation_seeking
        )
       
        # Get category and color
        category, color = get_mental_health_category(predicted_score)
       
        # Display results
        st.markdown("<div class='insight-box'>", unsafe_allow_html=True)
        st.write("### Your Predicted Mental Health Score")
       
        # Create a progress bar for score visualization
        st.markdown(f"""
        <div style="margin: 20px 0;">
            <div style="background-color: #f0f0f0; border-radius: 8px; height: 30px; width: 100%; position: relative;">
                <div style="background-color: {color}; width: {predicted_score}%; height: 100%; border-radius: 8px;">
                    <span style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); color: #000; font-weight: bold;">
                        {predicted_score:.1f}/100
                    </span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
       
        st.markdown(f"<h3 style='color: {color}; text-align: center; margin: 20px 0;'>Mental Health Category: {category}</h3>", unsafe_allow_html=True)
       
        # Get personalized recommendations from Gemini API
        st.write("### Your Personalized Recommendations")
        recommendations = get_gemini_recommendations(
            daily_usage,
            distraction_level,
            comparison_frequency,
            validation_seeking,
            predicted_score
        )
       
        for i, rec in enumerate(recommendations, 1):
            st.write(f"{i}. {rec}")
       
        st.markdown("</div>", unsafe_allow_html=True)
   
    with col2:
        # Show comparative charts (unchanged)
        if 'Daily Usage (minutes)' in df.columns:
            usage_data = df[df['Daily Usage (minutes)'].notna()]['Daily Usage (minutes)']
           
            fig = px.histogram(
                usage_data,
                nbins=20,
                title='How Your Usage Compares',
                opacity=0.7,
                color_discrete_sequence=[theme["chart_colors"][0]]
            )
           
            fig.add_vline(
                x=daily_usage,
                line_dash="dash",
                line_color=color,
                annotation_text="Your Usage",
                annotation_position="top"
            )
           
            avg_usage = usage_data.mean()
            fig.add_vline(
                x=avg_usage,
                line_dash="dot",
                line_color="gray",
                annotation_text="Average",
                annotation_position="bottom"
            )
           
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
       
        if 'Mental Health Score' in df.columns and 'Usage Category' in df.columns:
            mh_by_usage = df.groupby('Usage Category')['Mental Health Score'].mean().reset_index()
            usage_order = ['< 1 hour', '1-2 hours', '2-3 hours', '3-4 hours', '4+ hours']
            mh_by_usage['Usage Category'] = pd.Categorical(
                mh_by_usage['Usage Category'],
                categories=usage_order,
                ordered=True
            )
            mh_by_usage = mh_by_usage.sort_values('Usage Category')
           
            fig = px.bar(
                mh_by_usage,
                x='Usage Category',
                y='Mental Health Score',
                title='Mental Health Score by Usage',
                color='Mental Health Score',
                color_continuous_scale=theme["gradient_colors"]
            )
           
            user_category = None
            if daily_usage < 60:
                user_category = '< 1 hour'
            elif daily_usage < 120:
                user_category = '1-2 hours'
            elif daily_usage < 180:
                user_category = '2-3 hours'
            elif daily_usage < 240:
                user_category = '3-4 hours'
            else:
                user_category = '4+ hours'
           
            if user_category in mh_by_usage['Usage Category'].values:
                idx = mh_by_usage[mh_by_usage['Usage Category'] == user_category].index[0]
                fig.add_annotation(
                    x=user_category,
                    y=mh_by_usage.loc[idx, 'Mental Health Score'],
                    text="Your Usage Range",
                    showarrow=True,
                    arrowhead=1,
                    ax=0,
                    ay=-40
                )
           
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
       
        st.markdown("<div class='insight-box'>", unsafe_allow_html=True)
        st.write("### About This Predictor")
        st.write("This tool uses data from our survey of social media users to predict mental health impacts.")
        st.write("The model considers:")
        st.write("• Daily usage time")
        st.write("• Distraction patterns")
        st.write("• Social comparison behaviors")
        st.write("• Validation-seeking behaviors")
        st.write("These factors have shown strong correlations with mental health outcomes in research.")
        st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    show_dashboard()
