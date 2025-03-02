import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Global variable for CSV file path - change this to match your file location
CSV_FILE_PATH = "Data/train.csv"

# Page configuration
st.set_page_config(
    layout="wide", 
    page_title="Social Media & Mental Health Dashboard",
    page_icon="📱",
    initial_sidebar_state="expanded"
)

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
        chart_colors = ["#BB86FC", "#03DAC6", "#CF6679", "#FFAB40"]
        positive_color = "#03DAC6"   # Teal
        neutral_color = "#FFAB40"    # Amber
        negative_color = "#CF6679"   # Pink
    else:
        # Light theme colors
        header_color = "#6200EE"
        subheader_color = "#3700B3"
        text_color = "#000000"
        background_color = "#FFFFFF"
        box_background = "#F5F5F5"
        border_color = "#6200EE"
        chart_colors = ["#6200EE", "#03DAC6", "#B00020", "#FF6D00"]
        positive_color = "#03DAC6"   # Teal
        neutral_color = "#FF6D00"    # Orange
        negative_color = "#B00020"   # Red
    
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
</style>
""", unsafe_allow_html=True)

# Helper function to clean field names for display
def clean_field_name(field_name):
    # Replace underscores with spaces and capitalize each word
    return field_name.replace('_', ' ').replace('(', '(').replace(')', ')').title()

@st.cache_data(ttl=3600)  # Cache data for 1 hour
def load_and_process_data():
    """Load and preprocess the data from CSV file."""
    try:
        df = pd.read_csv(CSV_FILE_PATH)
        
        # Convert Age to numeric if it's stored as string but contains numbers
        try:
            df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
        except:
            pass  # Keep as is if conversion fails
        
        # Create age groups if Age is numeric
        if pd.api.types.is_numeric_dtype(df['Age']):
            bins = [0, 18, 25, 35, 50, 100]
            labels = ['Under 18', '18-24', '25-34', '35-49', '50+']
            df['Age Group'] = pd.cut(df['Age'], bins=bins, labels=labels)
        
        # Rename columns to be more readable
        df.columns = [col.replace('_', ' ').title() for col in df.columns]
        
        # Restore key columns with their original naming for code compatibility
        col_mapping = {
            'User Id': 'User_ID',
            'Daily Usage Time (Minutes)': 'Daily_Usage_Time (minutes)',
            'Posts Per Day': 'Posts_Per_Day',
            'Likes Received Per Day': 'Likes_Received_Per_Day',
            'Comments Received Per Day': 'Comments_Received_Per_Day',
            'Messages Sent Per Day': 'Messages_Sent_Per_Day',
            'Dominant Emotion': 'Dominant_Emotion'
        }
        
        df = df.rename(columns={v: k for k, v in col_mapping.items()})
        
        # Create engagement metrics
        df['Engagement Ratio'] = (df['Likes Received Per Day'] + df['Comments Received Per Day']) / (df['Posts Per Day'] + 0.001)
        
        # Categorize emotions (assumption - modify as needed)
        positive_emotions = ['Happy', 'Happiness', 'Excited', 'Content', 'Calm', 'Relaxed', 'Joy', 'Gratitude']
        negative_emotions = ['Sad', 'Sadness', 'Anxious', 'Anxiety', 'Anger', 'Angry', 'Depressed', 'Depression', 'Stressed', 'Stress', 'Lonely', 'Loneliness', 'Frustrated', 'Frustration']
        neutral_emotions = ['Neutral', 'Boredom', 'Bored']
        
        # Create emotion category column
        df['Emotion Category'] = 'Other'
        df.loc[df['Dominant Emotion'].str.lower().isin([e.lower() for e in positive_emotions]), 'Emotion Category'] = 'Positive'
        df.loc[df['Dominant Emotion'].str.lower().isin([e.lower() for e in negative_emotions]), 'Emotion Category'] = 'Negative'
        df.loc[df['Dominant Emotion'].str.lower().isin([e.lower() for e in neutral_emotions]), 'Emotion Category'] = 'Neutral'
        
        # Create usage time categories
        usage_bins = [0, 30, 60, 120, 240, 1000]
        usage_labels = ['< 30 min', '30-60 min', '1-2 hours', '2-4 hours', '4+ hours']
        df['Usage Category'] = pd.cut(df['Daily Usage Time (Minutes)'], bins=usage_bins, labels=usage_labels)
        
        # Create mental health score (inverse for negative emotions)
        # Higher score = better mental health
        df['Mental Health Score'] = 50  # Base score
        
        # Modify score based on emotion category
        df.loc[df['Emotion Category'] == 'Positive', 'Mental Health Score'] += 25
        df.loc[df['Emotion Category'] == 'Negative', 'Mental Health Score'] -= 25
        
        # Modify score based on usage time (heavy usage associated with lower scores)
        # Adjust these weights based on your data's actual correlation patterns
        usage_weights = {'< 30 min': 10, '30-60 min': 5, '1-2 hours': 0, 
                        '2-4 hours': -5, '4+ hours': -15}
        
        for category, weight in usage_weights.items():
            df.loc[df['Usage Category'] == category, 'Mental Health Score'] += weight
        
        # Ensure score stays in 0-100 range
        df['Mental Health Score'] = df['Mental Health Score'].clip(0, 100)
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def calculate_mental_health_score(screen_time, base_emotion="Neutral"):
    """Calculate predicted mental health score based on screen time."""
    # Load data to get average distributions
    df = load_and_process_data()
    
    if df is None:
        return 50  # Return default score if data can't be loaded
    
    # Determine usage category
    if screen_time < 30:
        usage_cat = '< 30 min'
    elif screen_time < 60:
        usage_cat = '30-60 min'
    elif screen_time < 120:
        usage_cat = '1-2 hours'
    elif screen_time < 240:
        usage_cat = '2-4 hours'
    else:
        usage_cat = '4+ hours'
    
    # Start with base score
    score = 50
    
    # Adjust score based on usage category
    usage_weights = {'< 30 min': 10, '30-60 min': 5, '1-2 hours': 0, 
                    '2-4 hours': -5, '4+ hours': -15}
    score += usage_weights.get(usage_cat, 0)
    
    # Get the dominant emotion categories for this usage level
    if len(df) > 0:
        filtered_df = df[df['Usage Category'] == usage_cat]
        if len(filtered_df) > 0:
            emotion_counts = filtered_df['Emotion Category'].value_counts(normalize=True)
            
            # Calculate probability-weighted score adjustment
            emotion_weights = {'Positive': 25, 'Neutral': 0, 'Negative': -25, 'Other': 0}
            for emotion, probability in emotion_counts.items():
                score += emotion_weights.get(emotion, 0) * probability
    
    # Ensure score stays in 0-100 range
    return max(0, min(100, score))

def get_mental_health_category(score):
    """Convert mental health score to category."""
    if score >= 80:
        return "Excellent", theme["positive_color"]
    elif score >= 60:
        return "Good", theme["positive_color"]
    elif score >= 40:
        return "Average", theme["neutral_color"]
    elif score >= 20:
        return "Concerning", theme["negative_color"]
    else:
        return "Poor", theme["negative_color"]

def show_dashboard():
    st.markdown("<h1 class='main-header'>📱 Social Media & Mental Health Dashboard</h1>", unsafe_allow_html=True)
    
    # Load data
    df = load_and_process_data()
    
    if df is None:
        st.error(f"Could not load data from {CSV_FILE_PATH}. Please check the file path and format.")
        return
    
    # Display file info
    st.info(f"Analyzing data from: **{CSV_FILE_PATH}** • {len(df)} users analyzed")
    
    # Dashboard tabs with improved styling
    tabs = st.tabs(["📊 Overview", "😊 Emotional Analysis", "🔄 Usage Correlations", "🔍 Advanced Insights", "🧠 Mental Health Predictor"])
    
    with tabs[0]:
        show_overview(df)
    
    with tabs[1]:
        show_emotional_analysis(df)
    
    with tabs[2]:
        show_correlations(df)
    
    with tabs[3]:
        show_advanced_insights(df)
    
    with tabs[4]:
        show_mental_health_predictor(df)
    
    # Footer
    st.markdown("---")
    st.markdown("AuraTrack: Social Media Mental Health Analysis • Dashboard v1.1", help="Developed for analyzing social media usage and mental health correlations")

def show_overview(df):
    st.markdown("<h2 class='sub-header'>Data Overview</h2>", unsafe_allow_html=True)
    
    # Key metrics in a row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-value'>{len(df)}</div>", unsafe_allow_html=True)
        st.markdown("<div class='metric-label'>Total Users</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        avg_usage = df['Daily Usage Time (Minutes)'].mean()
        st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-value'>{avg_usage:.1f}</div>", unsafe_allow_html=True)
        st.markdown("<div class='metric-label'>Avg. Daily Usage (min)</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col3:
        most_common_platform = df['Platform'].value_counts().index[0]
        platform_pct = (df['Platform'].value_counts()[0] / len(df) * 100)
        st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-value'>{most_common_platform}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-label'>Top Platform ({platform_pct:.1f}%)</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col4:
        most_common_emotion = df['Dominant Emotion'].value_counts().index[0]
        emotion_pct = (df['Dominant Emotion'].value_counts()[0] / len(df) * 100)
        st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-value'>{most_common_emotion}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-label'>Top Emotion ({emotion_pct:.1f}%)</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Data summary with expander
    with st.expander("View Data Preview"):
        st.write("### Raw Data Sample")
        st.dataframe(df.head())
        
        st.write("### Basic Statistics")
        numeric_df = df.select_dtypes(include=['float64', 'int64'])
        st.dataframe(numeric_df.describe())
    
    # Distribution of users
    st.markdown("<h3>User Demographics</h3>", unsafe_allow_html=True)
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Platform distribution
        platforms_count = df['Platform'].value_counts().reset_index()
        platforms_count.columns = ['Platform', 'Count']
        fig = px.pie(platforms_count, values='Count', names='Platform', 
                     title='Users by Platform',
                     color_discrete_sequence=theme["chart_colors"])
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Gender distribution
        gender_count = df['Gender'].value_counts().reset_index()
        gender_count.columns = ['Gender', 'Count']
        fig = px.pie(gender_count, values='Count', names='Gender', 
                     title='Users by Gender',
                     color_discrete_sequence=theme["chart_colors"])
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Age distribution if available
    if 'Age Group' in df.columns:
        st.markdown("<h3>Age Distribution</h3>", unsafe_allow_html=True)
        age_count = df['Age Group'].value_counts().reset_index()
        age_count.columns = ['Age Group', 'Count']
        
        fig = px.bar(age_count, x='Age Group', y='Count', 
                    title='Users by Age Group',
                    color='Count',
                    color_continuous_scale='Viridis')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Usage statistics
    st.markdown("<h3>Usage Patterns</h3>", unsafe_allow_html=True)
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Daily usage distribution
        fig = px.histogram(df, x='Daily Usage Time (Minutes)', 
                           nbins=20,
                           title='Distribution of Daily Usage Time',
                           color_discrete_sequence=[theme["chart_colors"][0]])
        fig.update_layout(bargap=0.1, height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Posts per day distribution
        fig = px.histogram(df, x='Posts Per Day', 
                           nbins=15,
                           title='Distribution of Posts Per Day',
                           color_discrete_sequence=[theme["chart_colors"][1]])
        fig.update_layout(bargap=0.1, height=400)
        st.plotly_chart(fig, use_container_width=True)

def show_emotional_analysis(df):
    st.markdown("<h2 class='sub-header'>Emotional Analysis</h2>", unsafe_allow_html=True)
    
    # Distribution of emotions
    emotion_count = df['Dominant Emotion'].value_counts().reset_index()
    emotion_count.columns = ['Emotion', 'Count']
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        fig = px.bar(emotion_count, x='Emotion', y='Count', 
                    title='Distribution of Dominant Emotions',
                    color='Emotion',
                    color_discrete_sequence=theme["chart_colors"])
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("<div class='insight-box'>", unsafe_allow_html=True)
        st.write("### Key Insights")
        
        # Find most common emotion
        most_common = emotion_count.iloc[0]['Emotion']
        st.write(f"• Most common emotion: **{most_common}**")
        
        # Calculate percentage of emotion categories
        if 'Emotion Category' in df.columns:
            emotion_cats = df['Emotion Category'].value_counts(normalize=True) * 100
            for cat, pct in emotion_cats.items():
                st.write(f"• **{cat}** emotions: **{pct:.1f}%** of users")
        
        # Find average usage time for most common emotions
        top_emotions = emotion_count.head(3)['Emotion'].tolist()
        for emotion in top_emotions:
            avg_time = df[df['Dominant Emotion'] == emotion]['Daily Usage Time (Minutes)'].mean()
            st.write(f"• Users feeling **{emotion}** use social media **{avg_time:.1f} min/day** on average")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Emotions by platform
    st.write("### Emotions by Platform")
    
    # Get top emotions and platforms for cleaner visualization
    top_emotions = emotion_count.head(6)['Emotion'].tolist()
    top_platforms = df['Platform'].value_counts().head(5).index.tolist()
    
    # Filter data for top emotions and platforms
    filtered_df = df[df['Dominant Emotion'].isin(top_emotions) & df['Platform'].isin(top_platforms)]
    
    # Create emotion by platform heatmap
    emotion_platform = pd.crosstab(filtered_df['Dominant Emotion'], filtered_df['Platform'])
    emotion_platform_pct = emotion_platform.div(emotion_platform.sum(axis=0), axis=1) * 100
    
    fig = px.imshow(emotion_platform_pct, 
                   labels=dict(x="Platform", y="Emotion", color="Percentage (%)"),
                   text_auto='.1f',
                   aspect="auto",
                   color_continuous_scale='Blues')
    fig.update_layout(height=450, 
                     xaxis_title="Platform", 
                     yaxis_title="Emotion",
                     coloraxis_colorbar=dict(title="Percentage (%)"))
    st.plotly_chart(fig, use_container_width=True)
    
    # Emotion categories by usage time
    if 'Emotion Category' in df.columns and 'Usage Category' in df.columns:
        st.write("### Emotion Categories by Usage Time")
        
        emotion_usage = pd.crosstab(df['Usage Category'], df['Emotion Category'])
        emotion_usage_pct = emotion_usage.div(emotion_usage.sum(axis=1), axis=0) * 100
        
        # Create a mapping of emotion categories to colors
        color_discrete_map = {
            'Positive': theme["positive_color"], 
            'Neutral': theme["neutral_color"], 
            'Negative': theme["negative_color"], 
            'Other': '#9E9E9E'
        }
        
        fig = px.bar(emotion_usage_pct.reset_index().melt(id_vars='Usage Category', var_name='Emotion Category', value_name='Percentage'),
                    x='Usage Category', 
                    y='Percentage', 
                    color='Emotion Category',
                    title='Emotion Categories by Usage Time',
                    barmode='stack',
                    color_discrete_map=color_discrete_map)
        
        fig.update_layout(height=450,
                         xaxis_title="Daily Usage Time",
                         yaxis_title="Percentage (%)")
        st.plotly_chart(fig, use_container_width=True)
        
        # Key insights about emotion categories and usage time
        st.markdown("<div class='insight-box'>", unsafe_allow_html=True)
        st.write("### Key Observations")
        
        # Find categories with highest usage time
        high_usage_emotions = df.groupby('Emotion Category')['Daily Usage Time (Minutes)'].mean().sort_values(ascending=False)
        
        for i, (cat, time) in enumerate(high_usage_emotions.items()):
            if i == 0:
                st.write(f"• Users with **{cat}** emotions spend the **most time** on social media (**{time:.1f} min/day**)")
            elif i == len(high_usage_emotions) - 1:
                st.write(f"• Users with **{cat}** emotions spend the **least time** on social media (**{time:.1f} min/day**)")
        
        # Check if there's a pattern in high usage category
        high_usage_df = df[df['Usage Category'] == '4+ hours']
        if len(high_usage_df) > 0:
            high_usage_top_emotion = high_usage_df['Dominant Emotion'].value_counts().index[0]
            high_usage_pct = high_usage_df['Dominant Emotion'].value_counts().iloc[0] / len(high_usage_df) * 100
            st.write(f"• Among heavy users (4+ hours), **{high_usage_pct:.1f}%** report feeling **{high_usage_top_emotion}**")
        
        st.markdown("</div>", unsafe_allow_html=True)

def show_correlations(df):
    st.markdown("<h2 class='sub-header'>Usage & Emotion Correlations</h2>", unsafe_allow_html=True)
    
    # Usage metrics by emotion
    st.write("### Social Media Usage by Dominant Emotion")
    
    usage_metrics = ['Daily Usage Time (Minutes)', 'Posts Per Day', 
                     'Likes Received Per Day', 'Comments Received Per Day', 
                     'Messages Sent Per Day']
    
    # Use clean names for display
    usage_metrics_display = [clean_field_name(metric) for metric in usage_metrics]
    
    # Create a mapping between display names and actual column names
    metrics_map = dict(zip(usage_metrics_display, usage_metrics))
    
    selected_metric_display = st.selectbox("Select usage metric to analyze:", usage_metrics_display)
    selected_metric = metrics_map[selected_metric_display]
    
    # Get top emotions for cleaner visualization
    top_emotions = df['Dominant Emotion'].value_counts().head(6).index.tolist()
    filtered_df = df[df['Dominant Emotion'].isin(top_emotions)]
    
    # Box plot of selected metric by emotion
    fig = px.box(filtered_df, 
                x='Dominant Emotion', 
                y=selected_metric, 
                color='Dominant Emotion',
                title=f'{selected_metric_display} by Dominant Emotion',
                color_discrete_sequence=theme["chart_colors"])
    
    fig.update_layout(height=500,
                     xaxis_title="Emotion",
                     yaxis_title=selected_metric_display,
                     showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    # Calculate means for each emotion
    emotion_means = filtered_df.groupby('Dominant Emotion')[selected_metric].mean().reset_index()
    emotion_means = emotion_means.sort_values(selected_metric, ascending=False)
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        # Bar chart of average metric by emotion
        fig = px.bar(emotion_means, 
                    x='Dominant Emotion', 
                    y=selected_metric,
                    color='Dominant Emotion',
                    title=f'Average {selected_metric_display} by Emotion',
                    color_discrete_sequence=theme["chart_colors"])
        
        fig.update_layout(height=400,
                         xaxis_title="Emotion",
                         yaxis_title=f"Average {selected_metric_display}",
                         showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("<div class='insight-box'>", unsafe_allow_html=True)
        st.write("### Statistical Insights")
        
        # Calculate ANOVA to test if there are significant differences between emotions
        try:
            groups = [filtered_df[filtered_df['Dominant Emotion'] == emotion][selected_metric].dropna() 
                     for emotion in filtered_df['Dominant Emotion'].unique() if len(filtered_df[filtered_df['Dominant Emotion'] == emotion]) > 0]
            
            if len(groups) > 1 and all(len(g) > 0 for g in groups):
                f_val, p_val = stats.f_oneway(*groups)
                
                st.write(f"• ANOVA p-value: **{p_val:.4f}**")
                if p_val < 0.05:
                    st.write("• There is a **statistically significant difference** in this metric across different emotional states.")
                else:
                    st.write("• There is **no statistically significant difference** in this metric across different emotional states.")
            else:
                st.write("• Not enough data in each group for statistical testing.")
        except Exception as e:
            st.write(f"• Unable to perform statistical test: {e}")
        
        # Find the emotion with highest average
        if not emotion_means.empty:
            highest_emotion = emotion_means.iloc[0]['Dominant Emotion']
            highest_value = emotion_means.iloc[0][selected_metric]
            st.write(f"• **{highest_emotion}** has the highest average **{selected_metric_display.lower()}** ({highest_value:.2f}).")
            
            # Find the emotion with lowest average
            lowest_emotion = emotion_means.iloc[-1]['Dominant Emotion']
            lowest_value = emotion_means.iloc[-1][selected_metric]
            st.write(f"• **{lowest_emotion}** has the lowest average **{selected_metric_display.lower()}** ({lowest_value:.2f}).")
            
            # Calculate percentage difference
            pct_diff = (highest_value - lowest_value) / lowest_value * 100
            st.write(f"• The difference between highest and lowest is **{pct_diff:.1f}%**.")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Correlation heatmap
    st.write("### Correlation Matrix")
    
    # Select only numeric columns for correlation
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    
    # Filter out ID columns and derived columns
    numeric_cols = [col for col in numeric_cols if 'ID' not in col and 'Ratio' not in col and 'Score' not in col]
    
    # Create correlation matrix
    corr_matrix = df[numeric_cols].corr().round(2)
    
    # Plot heatmap
    fig = px.imshow(corr_matrix,
                   text_auto='.2f',
                   aspect="auto",
                   color_continuous_scale='RdBu_r',
                   color_continuous_midpoint=0)
    
    fig.update_layout(height=550,
                     title="Correlation Matrix of Usage Metrics")
    st.plotly_chart(fig, use_container_width=True)
    
    # Highlight strongest correlations
    st.markdown("<div class='insight-box'>", unsafe_allow_html=True)
    st.write("### Key Correlations")
    
    # Remove self-correlations and find top 3 strongest correlations
    corr_data = corr_matrix.unstack()
    corr_data = corr_data[corr_data < 1.0]  # Remove self-correlations
    strongest_corrs = corr_data.abs().sort_values(ascending=False)[:3]
    
    for idx, corr_value in strongest_corrs.items():
        var1, var2 = idx
        direction = "positive" if corr_value > 0 else "negative"
        strength = "strong" if abs(corr_value) > 0.7 else "moderate"
        
        var1_label = clean_field_name(var1)
        var2_label = clean_field_name(var2)
        
        st.write(f"• **{var1_label}** and **{var2_label}** have a {strength} {direction} correlation (**{corr_value:.2f}**).")
    
    st.markdown("</div>", unsafe_allow_html=True)

def show_advanced_insights(df):
    st.markdown("<h2 class='sub-header'>Advanced Insights</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.write("### Usage Time vs. Emotion Category")
        
        if 'Emotion Category' in df.columns:
            # Create plot of average usage time by emotion category
            emotion_cat_usage = df.groupby('Emotion Category')['Daily Usage Time (Minutes)'].mean().reset_index()
            emotion_cat_usage = emotion_cat_usage.sort_values('Daily Usage Time (Minutes)', ascending=False)
            
            # Add count of users in each category
            emotion_cat_count = df['Emotion Category'].value_counts().reset_index()
            emotion_cat_count.columns = ['Emotion Category', 'Count']
            
            emotion_cat_usage = emotion_cat_usage.merge(emotion_cat_count, on='Emotion Category', how='left')
            emotion_cat_usage['Label'] = emotion_cat_usage['Emotion Category'] + ' (' + emotion_cat_usage['Count'].astype(str) + ' users)'
            
            # Create a mapping of emotion categories to colors
            color_discrete_map = {
                'Positive': theme["positive_color"], 
                'Neutral': theme["neutral_color"], 
                'Negative': theme["negative_color"], 
                'Other': '#9E9E9E'
            }
            
            fig = px.bar(emotion_cat_usage, 
                        x='Label', 
                        y='Daily Usage Time (Minutes)',
                        color='Emotion Category',
                        title='Average Daily Usage Time by Emotion Category',
                        color_discrete_map=color_discrete_map)
            
            fig.update_layout(height=400,
                             xaxis_title="Emotion Category",
                             yaxis_title="Avg. Daily Usage (minutes)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            # Calculate average usage time by emotion
            emotion_usage = df.groupby('Dominant Emotion')['Daily Usage Time (Minutes)'].mean().reset_index()
            emotion_usage = emotion_usage.sort_values('Daily Usage Time (Minutes)', ascending=False)
            
            fig = px.bar(emotion_usage, 
                        x='Dominant Emotion', 
                        y='Daily Usage Time (Minutes)',
                        color='Dominant Emotion',
                        title='Average Daily Usage Time by Emotion',
                        color_discrete_sequence=theme["chart_colors"])
            
            fig.update_layout(height=400,
                             xaxis_title="Emotion",
                             yaxis_title="Avg. Daily Usage (minutes)",
                             showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        # Extract insights
        st.markdown("<div class='insight-box'>", unsafe_allow_html=True)
        
        # Find emotions that have highest/lowest usage time
        highest_usage_emotion = df.groupby('Dominant Emotion')['Daily Usage Time (Minutes)'].mean().sort_values(ascending=False)
        lowest_usage_emotion = df.groupby('Dominant Emotion')['Daily Usage Time (Minutes)'].mean().sort_values()
        
        st.write("### Usage Time Insights")
        
        if not highest_usage_emotion.empty and not lowest_usage_emotion.empty:
            high_emotion = highest_usage_emotion.index[0]
            high_time = highest_usage_emotion.iloc[0]
            
            low_emotion = lowest_usage_emotion.index[0]
            low_time = lowest_usage_emotion.iloc[0]
            
            st.write(f"• Users feeling **{high_emotion}** spend the most time on social media (**{high_time:.1f} min/day**).")
            st.write(f"• Users feeling **{low_emotion}** spend the least time (**{low_time:.1f} min/day**).")
            
            # Calculate percentage difference
            pct_diff = (high_time - low_time) / low_time * 100
            st.write(f"• The difference between highest and lowest usage time is **{pct_diff:.1f}%**.")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.write("### Engagement vs. Emotion")
        
        if 'Engagement Ratio' in df.columns:
            # Group by emotion and calculate average engagement ratio
            engagement_by_emotion = df.groupby('Dominant Emotion')['Engagement Ratio'].mean().reset_index()
            engagement_by_emotion = engagement_by_emotion.sort_values('Engagement Ratio', ascending=False)
            
            fig = px.bar(engagement_by_emotion,
                        x='Dominant Emotion',
                        y='Engagement Ratio',
                        color='Dominant Emotion',
                        title='Average Engagement Ratio by Emotion',
                        color_discrete_sequence=theme["chart_colors"])
            
            fig.update_layout(height=400,
                             xaxis_title="Emotion",
                             yaxis_title="Avg. Engagement Ratio",
                             showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            
            # Provide insights
            st.markdown("<div class='insight-box'>", unsafe_allow_html=True)
            st.write("### Engagement Insights")
            
            if not engagement_by_emotion.empty:
                highest_engagement = engagement_by_emotion.iloc[0]['Dominant Emotion']
                highest_value = engagement_by_emotion.iloc[0]['Engagement Ratio']
                
                lowest_engagement = engagement_by_emotion.iloc[-1]['Dominant Emotion']
                lowest_value = engagement_by_emotion.iloc[-1]['Engagement Ratio']
                
                st.write(f"• **{highest_engagement}** users have the highest average engagement ratio (**{highest_value:.2f}**).")
                st.write(f"• **{lowest_engagement}** users have the lowest average engagement ratio (**{lowest_value:.2f}**).")
                
                # Add interpretation
                st.write("• Higher engagement ratio means users receive more interactions (likes/comments) per post.")
            
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            # If engagement ratio not available, show likes received by emotion
            likes_by_emotion = df.groupby('Dominant Emotion')['Likes Received Per Day'].mean().reset_index()
            likes_by_emotion = likes_by_emotion.sort_values('Likes Received Per Day', ascending=False)
            
            fig = px.bar(likes_by_emotion,
                        x='Dominant Emotion',
                        y='Likes Received Per Day',
                        color='Dominant Emotion',
                        title='Average Likes Received by Emotion',
                        color_discrete_sequence=theme["chart_colors"])
            
            fig.update_layout(height=400,
                             xaxis_title="Emotion",
                             yaxis_title="Avg. Likes Received Per Day",
                             showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    
    # Mental Health Score by Usage Time
    if 'Mental Health Score' in df.columns:
        st.write("### Mental Health Score by Usage Category")
        
        mh_score_by_usage = df.groupby('Usage Category')['Mental Health Score'].mean().reset_index()
        
        # Ensure the categories are in the correct order
        usage_order = ['< 30 min', '30-60 min', '1-2 hours', '2-4 hours', '4+ hours']
        mh_score_by_usage['Usage Category'] = pd.Categorical(
            mh_score_by_usage['Usage Category'], 
            categories=usage_order,
            ordered=True
        )
        mh_score_by_usage = mh_score_by_usage.sort_values('Usage Category')
        
        # Create gradient color based on score
        colors = [theme["positive_color"] if score > 60 else 
                 theme["neutral_color"] if score > 40 else 
                 theme["negative_color"] for score in mh_score_by_usage['Mental Health Score']]
        
        fig = px.bar(mh_score_by_usage,
                    x='Usage Category',
                    y='Mental Health Score',
                    title='Average Mental Health Score by Usage Time',
                    color='Mental Health Score',
                    color_continuous_scale=[theme["negative_color"], theme["neutral_color"], theme["positive_color"]],
                    range_color=[0, 100])
        
        fig.update_layout(height=450,
                         xaxis_title="Daily Usage Time",
                         yaxis_title="Avg. Mental Health Score")
        st.plotly_chart(fig, use_container_width=True)
        
        # Provide insights
        st.markdown("<div class='insight-box'>", unsafe_allow_html=True)
        st.write("### Mental Health Insights")
        
        # Calculate correlation between usage time and mental health score
        usage_mins = df['Daily Usage Time (Minutes)']
        mh_score = df['Mental Health Score']
        corr, p_value = stats.pearsonr(usage_mins, mh_score)
        
        st.write(f"• Correlation between usage time and mental health score: **{corr:.2f}**")
        
        if abs(corr) > 0.3:
            direction = "positive" if corr > 0 else "negative"
            st.write(f"• There is a **{direction} correlation** between usage time and mental health.")
            
            if corr < 0:
                st.write("• This suggests that **higher usage time is associated with lower mental health scores**.")
            else:
                st.write("• This suggests that **higher usage time is associated with higher mental health scores**.")
        else:
            st.write("• The correlation is relatively weak, suggesting that the relationship is complex.")
        
        # Find optimal usage time
        optimal_usage = mh_score_by_usage.loc[mh_score_by_usage['Mental Health Score'].idxmax()]
        st.write(f"• The optimal usage category appears to be **{optimal_usage['Usage Category']}** with an average mental health score of **{optimal_usage['Mental Health Score']:.1f}**.")
        
        st.markdown("</div>", unsafe_allow_html=True)

def show_mental_health_predictor(df):
    st.markdown("<h2 class='sub-header'>Mental Health Predictor</h2>", unsafe_allow_html=True)
    
    st.write("This tool estimates potential mental health impacts based on daily social media usage time.")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # User input for daily usage time
        screen_time = st.slider("Enter your daily social media usage (minutes):", 
                                min_value=0, max_value=480, value=120, step=15)
        
        # Calculate predicted mental health score
        predicted_score = calculate_mental_health_score(screen_time)
        category, color = get_mental_health_category(predicted_score)
        
        # Display results
        st.markdown("<div class='insight-box'>", unsafe_allow_html=True)
        st.write("### Predicted Mental Health Impact")
        
        # Display score with colored bar
        st.write(f"Based on **{screen_time} minutes** of daily social media usage:")
        
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
        
        # Recommendations based on score
        st.write("### Recommendations")
        
        if predicted_score >= 80:
            st.write("• Your social media usage appears well-balanced")
            st.write("• Continue maintaining healthy boundaries with technology")
            st.write("• Focus on quality interactions rather than quantity")
        elif predicted_score >= 60:
            st.write("• Your social media usage is generally healthy")
            st.write("• Consider periodic digital detox days")
            st.write("• Be mindful of content that triggers negative emotions")
        elif predicted_score >= 40:
            st.write("• Your usage is at a moderate level with some risk")
            st.write("• Try to reduce screen time by 15-30 minutes daily")
            st.write("• Schedule specific times for checking social media")
            st.write("• Use apps to monitor and limit your usage")
        elif predicted_score >= 20:
            st.write("• Your usage pattern shows risk for mental health")
            st.write("• Consider reducing usage by 30-50%")
            st.write("• Replace some social media time with offline activities")
            st.write("• Be selective about platforms and content")
        else:
            st.write("• Your usage pattern shows high risk for mental health")
            st.write("• Consider a significant reduction in screen time")
            st.write("• Seek support if you find it difficult to reduce usage")
            st.write("• Focus on real-world connections and activities")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        # Show usage distribution with marker for user input
        fig = px.histogram(df, x='Daily Usage Time (Minutes)', 
                          nbins=20,
                          title='Where You Stand Compared to Others',
                          opacity=0.7,
                          color_discrete_sequence=[theme["chart_colors"][0]])
        
        # Add vertical line for user input
        fig.add_vline(x=screen_time, 
                     line_dash="dash", 
                     line_color=color, 
                     annotation_text="Your Usage", 
                     annotation_position="top")
        
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
        
        # Show mental health score by usage category
        if 'Mental Health Score' in df.columns:
            mh_score_by_usage = df.groupby('Usage Category')['Mental Health Score'].mean().reset_index()
            
            # Ensure the categories are in the correct order
            usage_order = ['< 30 min', '30-60 min', '1-2 hours', '2-4 hours', '4+ hours']
            mh_score_by_usage['Usage Category'] = pd.Categorical(
                mh_score_by_usage['Usage Category'], 
                categories=usage_order,
                ordered=True
            )
            mh_score_by_usage = mh_score_by_usage.sort_values('Usage Category')
            
            fig = px.line(mh_score_by_usage,
                         x='Usage Category',
                         y='Mental Health Score',
                         title='Mental Health Score by Usage Category',
                         markers=True,
                         color_discrete_sequence=[theme["chart_colors"][1]])
            
            fig.update_layout(height=300,
                             xaxis_title="Daily Usage Time",
                             yaxis_title="Avg. Mental Health Score")
            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    show_dashboard()