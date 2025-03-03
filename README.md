Overview

This is an interactive Streamlit dashboard that analyzes the relationship between social media usage patterns and mental health indicators. The dashboard visualizes survey data to provide insights on how different social media behaviors correlate with mental wellbeing.
 
Features
- Comprehensive Analysis: Explore the connections between social media usage patterns and various mental health indicators
- Interactive Visualizations: Dynamic charts and graphs that update based on user selections
- User Segmentation: Automatic clustering of users into risk categories based on usage patterns and mental health metrics
- Correlation Analysis: Statistical breakdown of relationships between different variables
- Mental Health Predictor: Tool that estimates mental health scores based on social media habits
- Demographic Insights: Analysis of usage patterns and mental health across different demographic groups
- Personalized Recommendations: Targeted suggestions based on usage patterns and risk profiles

Dashboard Sections

- Overview: General demographics and high-level metrics about the survey participants
- Mental Health Indicators: Detailed analysis of mental health metrics and their distribution
- Social Media Usage: Breakdown of usage patterns, behaviors, and platforms
- Correlation Analysis: Statistical relationships between social media usage and mental health factors
- User Segments: Classification of users into risk categories with detailed profiles
- Mental Health Predictor: Interactive tool to estimate mental health impact based on usage patterns

Getting Started

Prerequisites

Python 3.7+
Streamlit 1.0+
Pandas
NumPy
Plotly
Matplotlib
Seaborn
SciPy
scikit-learn

Data Format
- Demographic information (age, gender, relationship status, occupation)
- Social media usage patterns (time spent, platforms used)
- Mental health indicators (distraction levels, anxiety, depression, sleep issues)
- Social comparison and validation-seeking behaviors

Customization
- Theme: The dashboard automatically adapts to Streamlit's light/dark theme settings
- Data Source: Change the CSV_FILE_PATH variable to use a different dataset
- Visualizations: Modify the chart colors and styles in the get_theme_specific_colors() function

Key Metrics
- Mental Health Score: Composite metric based on distraction, anxiety, comparison, and other indicators
- Social Media Impact Score: Measures the negative influence of social media on daily functioning
- Resilience Score: Indicates ability to maintain mental wellbeing despite social media usage
- User Segments: Categorizes users into Low Risk, Moderate Risk, and High Risk groups
