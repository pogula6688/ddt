import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from streamlit import session_state as state
from scipy import stats
import re
from datetime import datetime, timedelta, time
import openai
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os

# Set OpenAI API Key
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Check if the API key is set
if not openai.api_key:
    st.error("OpenAI API key is not set. Please set the OPENAI_API_KEY environment variable.")
    st.stop()

# Set page configuration for wider layout
st.set_page_config(page_title="Dynamic Digital Twin", layout="wide", page_icon="🛠")

# Custom CSS for styling
st.markdown("""
<style>
    :root {
        --primary-color: #4CAF50;
        --background-color: #F0F2F6;
        --secondary-background-color: #FFFFFF;
        --text-color: #000000;
    }
    body {
        background-color: var(--background-color);
        color: var(--text-color);
    }
    .title {
        text-align: center;
        font-size: 2.5em;
        color: var(--primary-color);
        margin-bottom: 20px;
        font-weight: bold;
    }
    .subtitle {
        text-align: center;
        font-size: 1.5em;
        color: #666;
        margin-bottom: 20px;
    }
    .header {
        text-align: center;
        font-size: 1.5em;
        margin-bottom: 10px;
        font-weight: bold;
    }
    .footer {
        text-align: center;
        font-size: 0.9em;
        color: gray;
        margin-top: 30px;
        padding: 10px;
        background-color: var(--secondary-background-color);
        border-top: 1px solid #ddd;
    }
    .sidebar .sidebar-content {
        background-color: var(--secondary-background-color);
        border-right: 1px solid #ddd;
    }
    .stButton > button {
        background-color: var(--primary-color);
        color: white;
        border: none;
        border-radius: 4px;
        padding: 10px 20px;
        font-size: 16px;
        cursor: pointer;
    }
</style>
""", unsafe_allow_html=True)

# Title of the app
st.markdown("<div class='title'>Dynamic Digital Twin Application for Retail Optimization</div>", unsafe_allow_html=True)

# Subtitle
st.markdown("<div class='subtitle'>Data Visualization, Machine Learning, and Monte Carlo Simulation</div>", unsafe_allow_html=True)

# Professional summary of the app
st.write("""
This application leverages advanced analytics and predictive capabilities to empower retail businesses. 
By utilizing AI-driven insights, it enables users to visualize data dynamically, identify trends, optimize performance metrics, and simulate future scenarios. 
With robust data processing and interactive visualizations, this tool facilitates informed decision-making and strategic planning in the retail sector.
Upload your dataset to uncover valuable insights and enhance your predictive analytics capabilities.
""")

# File uploader to upload any dataset
uploaded_file = st.file_uploader("Upload your dataset (CSV, XLS, XLSX)", type=['csv', 'xls', 'xlsx'])

# Common functions
def preprocess_data(df):
    """Fill missing values and process data for modeling."""
    df.columns = df.columns.str.lower()  # Convert all column names to lowercase
    for column in df.columns:
        if df[column].dtype == 'object':
            df[column].fillna(df[column].mode()[0], inplace=True)
        elif df[column].dtype == 'datetime64[ns]':
            df[column].fillna(df[column].mode()[0], inplace=True)
        else:
            df[column].fillna(df[column].mean(), inplace=True)
    return df

def is_datetime_related(column_name):
    """Check if a column name suggests datetime-related information using regex."""
    datetime_pattern = r'(year|month|day|date|time|quarter|week|hour|minute|second)'
    return re.search(datetime_pattern, column_name, re.IGNORECASE) is not None

def get_pure_numeric_columns(df):
    """Get numeric columns that are not datetime-related."""
    numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    return [col for col in numeric_columns if not is_datetime_related(col)]

def identify_time_based_columns(df):
    """Identify columns that may represent time-based data using regex."""
    time_based_columns = []
    for column in df.columns:
        if is_datetime_related(column):
            time_based_columns.append(column)
    return time_based_columns

def identify_numeric_kpis(df):
    """Identify numeric columns as potential KPIs, excluding datetime-related columns."""
    numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
    return [col for col in numeric_columns if not is_datetime_related(col)]

def get_relevant_columns(df):
    """Get numeric and categorical columns, excluding datetime-related ones and limiting categorical to <= 20 unique values."""
    numeric_columns = [col for col in df.select_dtypes(include=['number']).columns if not is_datetime_related(col)]
    categorical_columns = [col for col in df.select_dtypes(include=['object']).columns 
                           if not is_datetime_related(col) and df[col].nunique() <= 20]
    return numeric_columns, categorical_columns

def aggregate_data(df, group_col, agg_col, agg_func):
    """Aggregate the data based on the specified function."""
    if agg_func == 'Average':
        return df.groupby(group_col)[agg_col].mean().reset_index()
    elif agg_func == 'Total':
        return df.groupby(group_col)[agg_col].sum().reset_index()
    return df

def create_scorecard(df, agg_col, agg_func):
    """Create a scorecard for the specified aggregation function without grouping."""
    if agg_func == 'Total':
        aggregated_value = df[agg_col].sum()
    elif agg_func == 'Average':
        aggregated_value = df[agg_col].mean()
    elif agg_func == 'Count':
        aggregated_value = df[agg_col].count()
    else:
        return None

    return {"label": f"{agg_func} of {agg_col}", "value": f"{aggregated_value:,.2f}"}

def create_scorecard_chart(scorecards):
    """Create a scorecard chart from a list of scorecards."""
    fig = go.Figure()

    for i, scorecard in enumerate(scorecards):
        fig.add_trace(go.Indicator(
            mode="number+delta",
            value=float(scorecard["value"].replace(",", "")),
            title={'text': scorecard["label"], 'font': {'size': 18}},  
            number={'font': {'size': 36}}, 
            domain={'x': [0.2 * i, 0.2 * i + 0.2], 'y': [0.5, 1]}
        ))

    fig.update_layout(
        grid=dict(rows=1, columns=len(scorecards)),
        template="plotly_white",
        height=200
    )

    return fig

def create_doughnut_chart(df, x_col, y_col, threshold):
    """Create a doughnut chart and check if the number of unique x_col values exceeds the threshold."""
    unique_values = df[x_col].nunique()
    if unique_values > threshold:
        raise ValueError(f"The number of unique values in '{x_col}' exceeds the threshold of {threshold}. Doughnut chart not displayed for readability.")
    
    if x_col in df.select_dtypes(include=['object']).columns and y_col in df.select_dtypes(include=['number']).columns:
        aggregated_df = df.groupby(x_col)[y_col].sum().reset_index()
        fig = go.Figure(data=[go.Pie(labels=aggregated_df[x_col], values=aggregated_df[y_col], hole=0.4)])
        fig.update_layout(title=f"Doughnut Chart of {y_col} by {x_col}", hovermode="closest")
    elif x_col in df.select_dtypes(include=['object']).columns:
        aggregated_counts = df[x_col].value_counts().reset_index()
        aggregated_counts.columns = [x_col, 'Count']
        fig = go.Figure(data=[go.Pie(labels=aggregated_counts[x_col], values=aggregated_counts['Count'], hole=0.4)])
        fig.update_layout(title=f"Doughnut Chart of {x_col}", hovermode="closest")
    else:
        raise ValueError("Doughnut chart requires a categorical x-axis and a numeric aggregation value.")
    
    return fig

def create_bar_chart(df, x_col, y_col):
    """Create a bar chart and handle errors for non-numeric y-axis."""
    try:
        if df[y_col].dtype not in ['int64', 'float64']:
            raise ValueError(f"Bar chart requires a numeric y-axis. '{y_col}' is not numeric.")
        
        if x_col not in df.select_dtypes(include=['object']).columns:
            raise ValueError("Bar chart requires a categorical x-axis.")
        
        fig = px.bar(df, x=x_col, y=y_col, title=f"Bar Chart of {y_col} by {x_col}", hover_data=[x_col, y_col])
        fig.update_layout(title_font=dict(size=20), xaxis_title_font=dict(size=14), yaxis_title_font=dict(size=14))
        return fig
    except ValueError as e:
        st.error(f"Chart Error: {str(e)}")
        return None
    except Exception as e:
        st.error(f"Unexpected Error: {str(e)}")
        return None

def format_x_axis(fig, x_col, df):
    """Dynamically format the x-axis based on data type."""
    if df[x_col].dtype in ['int64', 'float64']:
        if df[x_col].between(1900, 2100).all():
            fig.update_xaxes(tickvals=df[x_col].unique(), ticktext=[str(int(year)) for year in df[x_col].unique()])
    elif df[x_col].dtype == 'datetime64[ns]':
        fig.update_xaxes(tickformat='%Y-%m-%d')
    return fig

def create_line_chart(df, x_col, y_col):
    """Create a line chart for datetime x-axis and numeric y-axis. Return None for non-numeric y-axis."""
    try:
        if x_col not in identify_datetime_columns(df):
            raise ValueError("Line chart requires a datetime column for the x-axis.")
        
        if y_col not in df.select_dtypes(include=['number']).columns:
            st.error(f"Chart Error: Line chart requires a numeric column for the y-axis. '{y_col}' is not numeric.")
            return None
        
        fig = px.line(df, x=x_col, y=y_col, title=f"Line Chart of {y_col} by {x_col}", hover_data=[x_col, y_col])
        fig = format_x_axis(fig, x_col, df)
        return fig
    except ValueError as e:
        st.error(f"Chart Error: {str(e)}")
        return None
    except Exception as e:
        st.error(f"Unexpected Error: {str(e)}")
        return None

def recommend_chart_columns(df):
    numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
    time_based_columns = identify_time_based_columns(df)
    
    numeric_y_columns = [col for col in numeric_columns if col not in time_based_columns]
    
    recommendations = []
    recommendations.append("Chart Column Recommendations:")
    
    if time_based_columns and numeric_y_columns:
        recommendations.append("Line Chart:")
        recommendations.append("  Possible X-axis columns (datetime):")
        recommendations.extend(f"    - {col}" for col in time_based_columns)
        recommendations.append("  Possible Y-axis columns (numeric, excluding datetime):")
        recommendations.extend(f"    - {col}" for col in numeric_y_columns)
    else:
        recommendations.append("Line Chart: No suitable columns found.")
    
    categorical_with_few_unique = [col for col in categorical_columns if df[col].nunique() <= 20]
    if categorical_with_few_unique and numeric_y_columns:
        recommendations.append("Doughnut Chart:")
        recommendations.append("  Possible X-axis columns (categorical with <= 20 unique values):")
        recommendations.extend(f"    - {col}" for col in categorical_with_few_unique)
        recommendations.append("  Possible Y-axis columns (numeric, excluding datetime):")
        recommendations.extend(f"    - {col}" for col in numeric_y_columns)
    else:
        recommendations.append("Doughnut Chart: No suitable columns found.")
    
    bar_x_columns = [col for col in categorical_columns if df[col].nunique() <= 20]
    if bar_x_columns and numeric_y_columns:
        recommendations.append("Bar Chart:")
        recommendations.append("  Possible X-axis columns (categorical with <= 20 unique values):")
        recommendations.extend(f"    - {col}" for col in bar_x_columns)
        recommendations.append("  Possible Y-axis columns (numeric, excluding datetime):")
        recommendations.extend(f"    - {col}" for col in numeric_y_columns)
    else:
        recommendations.append("Bar Chart: No suitable columns found.")
    
    return "\n".join(recommendations)

def identify_datetime_columns(df):
    """Identify datetime and datetime-like columns in the dataframe using regex."""
    datetime_pattern = r'(year|month|day|date|time|quarter|week|hour|minute|second)'
    datetime_columns = df.select_dtypes(include=['datetime64']).columns.tolist()
    datetime_like_columns = [col for col in df.columns if re.search(datetime_pattern, col, re.IGNORECASE)]
    return list(set(datetime_columns + datetime_like_columns))

def filter_dataframe(df, filter_col, filter_type, filter_value):
    """Filter dataframe based on datetime or datetime-like columns."""
    if pd.api.types.is_datetime64_any_dtype(df[filter_col]):
        if filter_type == 'Date Range':
            start_date, end_date = filter_value
            start_datetime = datetime.combine(start_date, time.min)
            end_datetime = datetime.combine(end_date, time.max)
            return df[(df[filter_col] >= start_datetime) & (df[filter_col] <= end_datetime)]
    else:
        return df[df[filter_col] == filter_value]
    return df

def update_scorecard(df, agg_col, agg_func):
    """Update scorecard based on filtered data."""
    try:
        if df[agg_col].dtype not in ['int64', 'float64']:
            raise ValueError(f"Column '{agg_col}' is not numerical. Only numerical columns are allowed for scorecards.")
        
        if is_datetime_related(agg_col):
            raise ValueError(f"Column '{agg_col}' appears to be datetime-related. Only pure numeric columns are allowed for scorecards.")
        
        if agg_func == 'Total':
            aggregated_value = df[agg_col].sum()
        elif agg_func == 'Average':
            aggregated_value = df[agg_col].mean()
        elif agg_func == 'Count':
            aggregated_value = df[agg_col].count()
        else:
            raise ValueError(f"Invalid aggregation function: {agg_func}")

        return {"label": f"{agg_func} of {agg_col}", "value": f"{aggregated_value:,.2f}"}
    except KeyError:
        raise KeyError(f"Column '{agg_col}' not found in the dataset.")
    except Exception as e:
        raise Exception(f"Error creating scorecard: {str(e)}")

def update_chart(df, x_col, y_col, chart_type, agg_func=None):
    """Update chart based on filtered data."""
    if agg_func:
        df = aggregate_data(df, x_col, y_col, agg_func)
    
    if chart_type == 'Bar':
        updated_chart = create_bar_chart(df, x_col, y_col)
    elif chart_type == 'Line':
        updated_chart = create_line_chart(df, x_col, y_col)
    elif chart_type == 'Doughnut':
        threshold = 20
        updated_chart = create_doughnut_chart(df, x_col, y_col, threshold)
    else:
        updated_chart = go.Figure()
    
    return updated_chart

def display_scorecards_and_charts(scorecards, generated_charts):
    if scorecards:
        st.subheader("Scorecards")
        scorecard_fig = create_scorecard_chart(scorecards)
        st.plotly_chart(scorecard_fig, use_container_width=True)

    st.subheader("Charts")
    
    for i in range(0, len(generated_charts), 3):
        row_charts = generated_charts[i:i+3]
        cols = st.columns(len(row_charts))
        
        for j, chart_info in enumerate(row_charts):
            with cols[j]:
                st.plotly_chart(chart_info["fig"], use_container_width=True)

def generate_gpt4_summary(scorecards, charts, df):
    """Generate an intelligent summary using GPT-4."""
    prompt = f"""
    Analyze the following scorecards and charts generated from a retail dataset:

    Scorecards:
    {', '.join([f"{s['label']}: {s['value']}" for s in scorecards])}

    Charts:
    {', '.join([f"{c['type']} chart of {c['y_col']} by {c['x_col']}" for c in charts])}

    Dataset summary:
    {df.describe().to_string()}

    Please provide a concise, intelligent summary of the insights that can be drawn from these visualizations and data. 
    Focus on key trends, patterns, and potential business implications for a retail business.
    Limit your response to 3-5 bullet points.
    """

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a retail analytics expert providing insights based on data visualizations."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=500,
        temperature=0.7
    )

    response = response.choices[0].message['content'].strip()

    lines = response.split('\n')
    
    processed_lines = []
    for line in lines:
        line = line.strip()
        if line:
            if re.match(r'^\d+\.?\s', line):
                if not line.endswith('.'):
                    line += '.'
                processed_lines.append(line)
            else:
                if processed_lines:
                    processed_lines[-1] += ' ' + line
                else:
                    processed_lines.append(line)
    
    final_response = '\n'.join(processed_lines)
    
    return final_response

def generate_insights(data_df, num_scorecards, num_charts):
    global scorecards, generated_charts
    scorecards = []
    generated_charts = []

    for i in range(num_scorecards):
        agg_col = st.session_state[f"agg_col_{i}"]
        agg_func = st.session_state[f"agg_func_{i}"]
        if agg_col and agg_func:
            try:
                scorecard = update_scorecard(data_df, agg_col, agg_func)
                if scorecard:
                    scorecards.append(scorecard)
            except ValueError as ve:
                st.error(f"Scorecard {i+1} Error: {str(ve)}")
            except KeyError as ke:
                st.error(f"Scorecard {i+1} Error: {str(ke)}")
            except Exception as e:
                st.error(f"Scorecard {i+1} Error: An unexpected error occurred - {str(e)}")

    for i in range(num_charts):
        x_col = st.session_state[f"x_col_{i}"]
        y_col = st.session_state[f"y_col_{i}"]
        agg_func = st.session_state[f"agg_func_chart_{i}"]
        chart_type = st.session_state[f"chart_type_{i}"]

        if x_col and y_col and chart_type:
            try:
                fig = update_chart(data_df, x_col, y_col, chart_type, agg_func)
                if fig is not None: 
                    generated_charts.append({"fig": fig, "type": chart_type, "x_col": x_col, "y_col": y_col, "agg_func": agg_func})
            except Exception as e:
                st.error(f"Error generating chart: {str(e)}")

    display_scorecards_and_charts(scorecards, generated_charts)

    with st.spinner("Generating insights..."):
        insights = generate_gpt4_summary(scorecards, generated_charts, data_df)
        st.session_state.insights = insights
        st.subheader("Intelligent Insights")
        st.write(insights)

def prepare_advanced_context(df, scorecards=None, generated_charts=None, insights=None):
    data_summary = analyze_dataset(df)
    correlation_analysis = perform_correlation_analysis(df)
    outlier_analysis = perform_outlier_analysis(df)
    trend_analysis = perform_trend_analysis(df)
    
    context = f"""
    Advanced Data Analysis:
    {data_summary}
    
    Correlation Analysis:
    {correlation_analysis}
    
    Outlier Analysis:
    {outlier_analysis}
    
    Trend Analysis:
    {trend_analysis}
    """

    if scorecards:
        scorecard_info = ', '.join([f"{s['label']}: {s['value']}" for s in scorecards])
        context += f"\nScorecards:\n{scorecard_info}"

    if generated_charts:
        chart_info = ', '.join([f"{c['type']} chart of {c['y_col']} by {c['x_col']}" for c in generated_charts])
        context += f"\nCharts:\n{chart_info}"

    if insights:
        context += f"\nInsights:\n{insights}"
    else:
        context += "\nInsights: No insights generated yet."

    return context

def analyze_dataset(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    summary = f"""
    Dataset Overview:
    - Number of rows: {len(df)}
    - Number of columns: {len(df.columns)}
    - Numeric columns: {', '.join(numeric_cols)}
    - Categorical columns: {', '.join(categorical_cols)}
    
    Statistical Summary of Numeric Columns:
    {df[numeric_cols].describe().to_string()}
    
    Categorical Columns Summary:
    {df[categorical_cols].describe().to_string()}
    """
    return summary

def perform_correlation_analysis(df):
    numeric_df = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr()
    
    high_correlations = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > 0.7:
                high_correlations.append(f"{corr_matrix.columns[i]} and {corr_matrix.columns[j]}: {corr_matrix.iloc[i, j]:.2f}")
    
    correlation_summary = f"""
    High Correlations (|r| > 0.7):
    {', '.join(high_correlations)}
    """
    return correlation_summary

def perform_outlier_analysis(df):
    numeric_df = df.select_dtypes(include=[np.number])
    outliers = {}
    
    for col in numeric_df.columns:
        Q1 = numeric_df[col].quantile(0.25)
        Q3 = numeric_df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers[col] = ((numeric_df[col] < lower_bound) | (numeric_df[col] > upper_bound)).sum()
    
    outlier_summary = "Outlier Count per Numeric Column:\n"
    outlier_summary += '\n'.join([f"{col}: {count}" for col, count in outliers.items()])
    return outlier_summary

def perform_trend_analysis(df):
    time_cols = df.select_dtypes(include=['datetime64']).columns
    if len(time_cols) == 0:
        return "No datetime columns found for trend analysis."
    
    time_col = time_cols[0]
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    trends = []
    for col in numeric_cols:
        trend, _ = stats.linregress(df.index, df[col])[:2]
        if abs(trend) > 0.1:
            direction = "increasing" if trend > 0 else "decreasing"
            trends.append(f"{col} shows a {direction} trend")
    
    trend_summary = "Observed Trends:\n"
    trend_summary += '\n'.join(trends)
    return trend_summary

def ask_gpt4(question, context):
    prompt = f"""
    As an advanced retail analytics expert with deep knowledge of data analysis, statistics, and business intelligence, please address the following question based on the provided context about a dataset, its visualizations, insights, and advanced analysis.

    Context:
    {context}

    Question: {question}

    Please provide a comprehensive and insightful answer considering the following:
    1. Relevant statistical information from the dataset
    2. Correlations between variables pertinent to the question
    3. Observed trends or patterns in the data
    4. Potential business implications or recommendations
    5. Limitations of the analysis or areas that might require further investigation

    Your response should be detailed yet concise, suitable for a business audience. If the question cannot be fully answered based on the available information, please indicate this clearly and provide the best possible insights given the limitations.
    """

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an advanced retail analytics expert providing in-depth insights based on comprehensive data analysis."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=800,
            temperature=0.6
        )
        answer = response.choices[0].message['content'].strip()
        
        disclaimer = "\n\nPlease note: This analysis is based on the available data and should be interpreted in conjunction with industry expertise and additional relevant information."
        return answer + disclaimer
    
    except openai.error.InvalidRequestError as e:
        raise ValueError("The question is too complex for the current analysis capabilities.")
    except openai.error.RateLimitError:
        raise RuntimeError("The service is currently experiencing high demand. Please try again later.")
    except Exception as e:
        raise RuntimeError(f"An unexpected error occurred: {str(e)}")

def get_kpi_insights(kpis, df):
    """Get insights for the identified KPIs using LLM."""
    columns = df.columns.tolist()
    column_types = df.dtypes.tolist()
    column_info = [f"{col} ({dtype})" for col, dtype in zip(columns, column_types)]
    
    prompt = f"""
    Given the following columns from a retail dataset:
    {', '.join(column_info)}

    And the following columns identified as potential KPIs:
    {', '.join(kpis)}

    Please provide insights on why each of these numeric columns could be important as a KPI for retail optimization.
    Format your response as a numbered list, with each KPI and its explanation on a new line.
    Focus only on the provided numeric columns as KPIs.
    """

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a retail analytics expert providing insights on key performance indicators."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=500,
        temperature=0.7
    )

    kpi_insights = response.choices[0].message['content'].strip()
    return kpi_insights

def create_feature_importance_chart(importances, feature_names, target_variable):
    """Create a feature importance chart using Plotly."""
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    fig = px.bar(importance_df, x='Feature', y='Importance', title='Feature Importance')
    st.write("### How to Interpret the Feature Importance Chart")
    st.write(f"""
    - To **increase** the KPI {target_variable}, focus on the features with **positive** importance values.
    - To **decrease** the KPI {target_variable}, focus on the features with **negative** importance values.
    """)
    fig.update_layout(xaxis_title='Features', yaxis_title='Importance', title_font=dict(size=20))
    return fig

def filter_high_cardinality_features(df, threshold=15):
    """Filter out features with more than 'threshold' unique values."""
    return [col for col in df.columns if df[col].nunique() <= threshold]

def generate_llm_recommendations(feature_importances, target_variable):
    """Generate recommendations using OpenAI's LLM with sentence completion and numbered formatting."""
    prompt = f"""
    Based on the following feature importances, provide recommendations for optimizing the target variable '{target_variable}'. 
    The feature importance values are listed below:

    {feature_importances}

    Please provide 3-5 actionable recommendations to improve the KPI '{target_variable}'. 
    Number each recommendation.
    Consider both positive and negative importance scores in your recommendations.
    Explain how each important feature might affect '{target_variable}' and suggest specific actions to optimize it.
    Ensure all your recommendations are complete sentences and end with a period.
    Start each numbered recommendation on a new line.
    """
    
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a retail optimization expert providing data-driven recommendations."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=500,
        temperature=0.7
    )
    
    recommendations = response.choices[0].message['content'].strip()
    
    lines = recommendations.split('\n')
    
    processed_lines = []
    for line in lines:
        line = line.strip()
        if line:
            if re.match(r'^\d+\.?\s', line):
                if not line.endswith('.'):
                    line += '.'
                processed_lines.append(line)
            else:
                if processed_lines:
                    processed_lines[-1] += ' ' + line
                else:
                    processed_lines.append(line)
    
    final_recommendations = '\n'.join(processed_lines)
    
    return final_recommendations

def validate_simulation_results(df, target_variable, best_scenario, best_result):
    """Validate the simulation results using a holdout set."""
    # Split the data into training and validation sets
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # Calculate the expected change based on the best scenario
    expected_change = best_result - train_df[target_variable].mean()
    
    # Apply the scenario to the validation set
    for col, value in best_scenario.items():
        if col in val_df.columns and col != target_variable:
            if isinstance(value, (int, float)):
                val_df[col] = value
            else:
                val_df[col] = value
    
    # Calculate the actual change in the validation set
    actual_change = val_df[target_variable].mean() - train_df[target_variable].mean()
    
    # Calculate validation metrics
    mae = mean_absolute_error([expected_change], [actual_change])
    mse = mean_squared_error([expected_change], [actual_change])
    rmse = np.sqrt(mse)
    r2 = r2_score(val_df[target_variable], [best_result] * len(val_df))
    
    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "Expected Change": expected_change,
        "Actual Change": actual_change
    }

def simulate_target(df, target_variable, scenario, numeric_columns, categorical_columns):
    """Simulate the impact of a scenario on the target variable, including categorical variables."""
    numeric_impact = sum(df[var].corr(df[target_variable]) * (scenario[var] - df[var].mean()) for var in numeric_columns if var != target_variable)
    
    categorical_impact = 0
    for col in categorical_columns:
        if col in scenario:
            category = scenario[col]
            category_mean = df[df[col] == category][target_variable].mean()
            overall_mean = df[target_variable].mean()
            categorical_impact += category_mean - overall_mean
    
    return df[target_variable].mean() + numeric_impact + categorical_impact

def run_monte_carlo_simulation(df, target_variable, target_increase_percent, n_simulations=10000):
    """Run Monte Carlo simulation to optimize the target KPI, including filtered categorical variables."""
    base_value = df[target_variable].mean()
    target_value = base_value * (1 + target_increase_percent / 100)
    
    numeric_columns, categorical_columns = get_relevant_columns(df)
    
    # Create a mapping of categorical columns to their unique values
    categorical_mappings = {col: df[col].unique().tolist() for col in categorical_columns}
    
    best_scenario = None
    best_result = base_value
    
    for _ in range(n_simulations):
        scenario = {}
        for col in numeric_columns:
            if col != target_variable:
                # Randomly adjust each numeric variable between -50% to +100%
                adjustment = np.random.uniform(-0.5, 1.0)
                scenario[col] = df[col].mean() * (1 + adjustment)
        
        # Randomly select one category for each categorical variable
        for col in categorical_columns:
            selected_category = np.random.choice(categorical_mappings[col])
            scenario[col] = selected_category
        
        # Simulate the impact on the target variable
        simulated_value = simulate_target(df, target_variable, scenario, numeric_columns, categorical_columns)
        
        if simulated_value > best_result and simulated_value <= target_value:
            best_scenario = scenario
            best_result = simulated_value
    
    # Validate the simulation results
    validation_results = validate_simulation_results(df, target_variable, best_scenario, best_result)
    
    return best_scenario, best_result, categorical_mappings, validation_results

def print_target_variable_relationships(df, target_variable, numeric_variables):
    """Print the Pearson correlation between the target variable and numeric variables."""
    correlations = df[[target_variable] + numeric_variables].corr(method='pearson')[target_variable].drop(target_variable)
    st.write("### Relationships with Target Variable")
    st.write(f"Pearson Correlation coefficients between {target_variable} and numeric variables:")
    for var, corr in correlations.items():
        st.write(f"{var}: {corr:.4f}")
    st.write("Note: Pearson Correlation ranges from -1 to 1. Closer to 1 or -1 indicates stronger linear relationship.")
    st.write("Interpretation guide:")
    st.write("- 0.00-0.19: very weak")
    st.write("- 0.20-0.39: weak")
    st.write("- 0.40-0.59: moderate")
    st.write("- 0.60-0.79: strong")
    st.write("- 0.80-1.00: very strong")
    st.write("(Use the absolute value of the coefficient for this interpretation)")

def optimization_analysis(df, target_variable, target_increase_percent):
    """Perform optimization analysis using Monte Carlo simulation, including filtered categorical variables."""
    best_scenario, best_result, categorical_mappings, validation_results = run_monte_carlo_simulation(df, target_variable, target_increase_percent)
    
    optimizations = {}
    numeric_columns, categorical_columns = get_relevant_columns(df)
    
    for var in numeric_columns:
        if var != target_variable:
            current_value = df[var].mean()
            new_value = best_scenario[var]
            change_percent = ((new_value - current_value) / current_value) * 100
            optimizations[var] = {
                "type": "numeric",
                "direction": "increase" if change_percent > 0 else "decrease",
                "change_percent": abs(change_percent),
                "new_value": new_value
            }
    
    for col in categorical_columns:
        optimizations[col] = {
            "type": "categorical",
            "recommended_category": best_scenario[col]
        }
    
    return optimizations, best_result, validation_results

# Main app logic
if uploaded_file is not None:
    # Read the data
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith('.xls') or uploaded_file.name.endswith('.xlsx'):
        df = pd.read_excel(uploaded_file)

    # Preprocess the data
    df = preprocess_data(df)

    # Display dataset preview
    st.write("### Dataset Preview")
    st.dataframe(df.head())

    # Create tabs for different analyses
    tab1, tab2, tab3 = st.tabs([
    "📊 Data Visualization",
    "🧠 Machine Learning Optimization",
    "📈 Monte Carlo Simulation"
    ])

    with tab1:
        st.header("Dynamic Data Visualization")
        
        # Display recommendations
        recommendations = recommend_chart_columns(df)
        st.subheader("Chart Recommendations")
        st.text(recommendations)

        # List of all columns in the dataframe 
        columns = df.columns.tolist()
    
        num_scorecards = st.sidebar.number_input("Number of Scorecards", min_value=0, max_value=10, value=1, step=1)
        num_charts = st.sidebar.number_input("Number of Charts", min_value=1, max_value=10, value=1, step=1)

        for i in range(num_scorecards):
            st.sidebar.subheader(f"Scorecard {i + 1} Configuration")
            
            pure_numeric_columns = get_pure_numeric_columns(df)
            agg_col = st.sidebar.selectbox(f"Select Aggregation column for Scorecard {i + 1}:", 
                                        options=[""] + pure_numeric_columns, 
                                        key=f"agg_col_{i}")
            agg_func = st.sidebar.selectbox(f"Select Aggregation Function for Scorecard {i + 1}:", 
                                            [""] + ["Total", "Average", "Count"], 
                                            key=f"agg_func_{i}")

        for i in range(num_charts):
            st.sidebar.subheader(f"Chart {i + 1} Configuration")
            
            x_col = st.sidebar.selectbox(f"Select X-axis column for Chart {i + 1}:", options=[""] + columns, key=f"x_col_{i}")
            y_col = st.sidebar.selectbox(f"Select Y-axis column for Chart {i + 1}:", options=[""] + columns, key=f"y_col_{i}")
            agg_func = st.sidebar.selectbox(f"Select aggregation function for Chart {i + 1}:", [""] + ["Average", "Total"], key=f"agg_func_chart_{i}")
            chart_type = st.sidebar.selectbox(f"Select Chart Type for Chart {i + 1}:", [""] + ["Bar", "Line", "Doughnut"], key=f"chart_type_{i}")

        # Datetime Filtering (Optional)
        st.subheader("Datetime Filtering (Optional)")
        datetime_cols = identify_datetime_columns(df)
        filtered_df = df.copy()
        
        if datetime_cols:
            use_filtering = st.checkbox("Use datetime filtering")
            if use_filtering:
                selected_datetime_col = st.selectbox("Select column for filtering", datetime_cols)
                
                if pd.api.types.is_datetime64_any_dtype(df[selected_datetime_col]):
                    min_date = df[selected_datetime_col].min().date()
                    max_date = df[selected_datetime_col].max().date()
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        start_date = st.date_input("Start Date", min_date, min_value=min_date, max_value=max_date)
                    with col2:
                        end_date = st.date_input("End Date", max_date, min_value=min_date, max_value=max_date)
                    
                    filter_type = 'Date Range'
                    filter_value = (start_date, end_date)
                else:
                    unique_values = sorted(df[selected_datetime_col].unique())
                    filter_value = st.selectbox(f"Select {selected_datetime_col}", unique_values)
                    filter_type = 'Value'

                if st.button("Apply Filter and Generate Insights", key="filter_insights_button"):
                    filtered_df = filter_dataframe(df, selected_datetime_col, filter_type, filter_value)
                    st.success("Filter applied successfully!")
                    
                    generate_insights(filtered_df, num_scorecards, num_charts)
            else:
                st.info("No filter applied. Using full dataset.")
                
                if st.button("Generate Insights", key="full_dataset_insights_button"):
                    generate_insights(df, num_scorecards, num_charts)
        else:
            st.warning("No datetime or datetime-like columns found for filtering.")
            
            if st.button("Generate Insights"):
                generate_insights(df, num_scorecards, num_charts)

        # Q&A functionality
        
        with st.expander("Ask Questions", expanded=False):
            if 'user_question' not in state:
                state.user_question = ""
            if 'gpt4_answer' not in state:
                state.gpt4_answer = ""

            st.write("""Ask questions about the data, trends, correlations, or business implications. 
                    Our AI-powered analytics engine will provide insights based on the available information.
                    """)

            user_question = st.text_input("Enter your question:", value=state.user_question)

            if user_question != state.user_question:
                state.user_question = user_question
                state.gpt4_answer = ""

            if st.button("Generate Answer", key="qa_insights_button") and user_question:
                with st.spinner("Analyzing data and generating insights..."):
                    try:
                        # Use global variables if they exist, otherwise pass None
                        scorecards = globals().get('scorecards', None)
                        generated_charts = globals().get('generated_charts', None)
                        insights = state.insights if 'insights' in state else None

                        advanced_context = prepare_advanced_context(df, scorecards, generated_charts, insights)
                        answer = ask_gpt4(user_question, advanced_context)
                        state.gpt4_answer = answer
                    except openai.error.InvalidRequestError as e:
                        st.error("We apologize, but the question appears to be too complex for our current analysis capabilities. Please try rephrasing or simplifying your question.")
                        state.gpt4_answer = None
                    except openai.error.RateLimitError:
                        st.error("We're experiencing high demand. Please try again in a few moments.")
                        state.gpt4_answer = None
                    except Exception as e:
                        st.error(f"An unexpected error occurred: {str(e)}. Please try again or contact support if the issue persists.")
                        state.gpt4_answer = None

            if state.gpt4_answer:
                st.subheader("Analysis:")
                st.markdown(state.gpt4_answer)
                st.info("Note: This analysis is based on the available data and should be used in conjunction with domain expertise and other relevant information.")
            elif state.gpt4_answer is None and user_question:
                st.warning("We couldn't generate insights for your question. Please try rephrasing or ask a different question.") 

    with tab2:
        st.header("Machine Learning for KPI Optimization")
        
        # Identify numeric KPIs, excluding datetime-related columns
        kpis = identify_numeric_kpis(df)
        
        if kpis:
            # Get insights for the identified KPIs
            kpi_insights = get_kpi_insights(kpis, df)
            
            st.write("### Key Performance Indicators (KPIs) for Optimization")
            st.write("The following columns have been identified as potential KPIs:")
            st.write(kpi_insights)
            
            # Dropdown for target variable selection
            target_variable = st.selectbox("Select KPI for Machine Learning Optimization", kpis)

            if target_variable:
                if st.button("Run Machine Learning Analysis"):
                    with st.spinner("Performing Machine Learning analysis..."):
                        # Separate features and target
                        X = df.drop(columns=[target_variable])
                        y = df[target_variable]

                        # Filter out high cardinality features
                        valid_features = filter_high_cardinality_features(X)
                        X = X[valid_features]

                        # Identify categorical columns
                        categorical_features = X.select_dtypes(include=['object']).columns

                        # Preprocessing pipeline
                        preprocessor = ColumnTransformer(
                            transformers=[
                                ('num', SimpleImputer(strategy='mean'), X.select_dtypes(include=['number']).columns),
                                ('cat', Pipeline(steps=[
                                    ('imputer', SimpleImputer(strategy='most_frequent')),
                                    ('onehot', OneHotEncoder(handle_unknown='ignore'))
                                ]), categorical_features)
                            ])

                        # Create and train model pipeline
                        model_pipeline = Pipeline(steps=[
                            ('preprocessor', preprocessor),
                            ('regressor', LinearRegression())
                        ])

                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                        model_pipeline.fit(X_train, y_train)


                        # Make predictions on test set
                        y_pred = model_pipeline.predict(X_test)

                        # Calculate evaluation metrics
                        mae = mean_absolute_error(y_test, y_pred)
                        mse = mean_squared_error(y_test, y_pred)
                        rmse = np.sqrt(mse)
                        r2 = r2_score(y_test, y_pred)

                         # Print evaluation metrics
                        st.write("### Model Evaluation Metrics")
                        st.write(f"Mean Absolute Error: {mae:.4f}")
                        st.write(f"Mean Squared Error: {mse:.4f}")
                        st.write(f"Root Mean Squared Error: {rmse:.4f}")
                        st.write(f"R-squared: {r2:.4f}")
                       
                        # Interpret evaluation results
                        st.write("### Interpretation of Evaluation Results")

                        # Determine model performance category
                        if r2 > 0.7:
                            performance_category = "good predictive power"
                            performance_suggestion = "The model seems to be performing well."
                        elif r2 > 0.5:
                            performance_category = "moderate predictive power"
                            performance_suggestion = "There might be room for improvement. Consider feature engineering or trying other algorithms."
                        else:
                            performance_category = "need for improvement"
                            performance_suggestion = "The model's performance is not satisfactory. Consider feature engineering, gathering more data, or trying other algorithms."

                        st.write(f"""
                        - Mean Absolute Error (MAE): On average, our predictions are off by {mae:.4f} units of {target_variable}.
                        - Root Mean Squared Error (RMSE): This gives more weight to large errors. Our model's RMSE is {rmse:.4f}.
                        - R-squared: Our model explains {r2:.2%} of the variance in the {target_variable}.

                        Interpretation:
                        1. The MAE and RMSE are in the same units as {target_variable}. Lower values indicate better performance.
                        2. R-squared ranges from 0 to 1, with 1 indicating perfect prediction. Our value of {r2:.4f} suggests that the model has {performance_category}.
                        3. {performance_suggestion}
                        """)


                        # Extract feature names after one-hot encoding
                        numerical_features = X.select_dtypes(include=['number']).columns
                        categorical_encoded_features = list(model_pipeline.named_steps['preprocessor']
                                                            .named_transformers_['cat']
                                                            .named_steps['onehot']
                                                            .get_feature_names_out(categorical_features))
                        feature_names = list(numerical_features) + categorical_encoded_features

                        # Ensure all feature names are lowercase
                        feature_names = [name.lower() for name in feature_names]

                        # Extract feature importances (for LinearRegression, use coefficients)
                        importances = model_pipeline.named_steps['regressor'].coef_

                        # Ensure importances and feature_names have the same length
                        if len(importances) != len(feature_names):
                            st.error("Mismatch in lengths of feature importances and feature names.")
                        else:
                            # Create and display feature importance chart
                            st.plotly_chart(create_feature_importance_chart(importances, feature_names, target_variable), use_container_width=True)

                            # Generate recommendations using LLM
                            feature_importances = dict(zip(feature_names, importances))
                            recommendations = generate_llm_recommendations(feature_importances, target_variable)

                            st.write(f"### Intelligent Recommendations for Optimizing '{target_variable}'")
                            st.write(recommendations)
        else:
            st.warning("No suitable columns found in the dataset to use as KPIs.")
                
                     
    with tab3:
        st.header("KPI Optimization with Monte Carlo Simulation")
        
        # Identify numeric KPIs
        kpis = identify_numeric_kpis(df)

        if kpis:
            target_variable = st.selectbox("Select the KPI to optimize", kpis)
            target_increase = st.slider(f"Select target increase percentage for {target_variable}", 
                                        min_value=1, max_value=100, value=10, step=1)

            if st.button("Run Optimization Analysis"):
                with st.spinner("Performing Monte Carlo simulation for optimization..."):
                    optimizations, best_result, validation_results = optimization_analysis(df, target_variable, target_increase)
                    
                    numeric_variables = [var for var, opt in optimizations.items() if opt["type"] == "numeric"]
                    print_target_variable_relationships(df, target_variable, numeric_variables)
                                    
                    st.write(f"### Optimization Results for {target_variable}")
                    st.write(f"Current average {target_variable}: {df[target_variable].mean():.2f}")
                    st.write(f"Best simulated {target_variable}: {best_result:.2f}")
                    st.write(f"Target {target_variable}: {df[target_variable].mean() * (1 + target_increase / 100):.2f}")
                    
                    numeric_optimizations = []
                    categorical_optimizations = []
                    
                    for var, opt in optimizations.items():
                        if opt["type"] == "numeric":
                            numeric_optimizations.append({
                                "Variable": var,
                                "Current Value": f"{df[var].mean():.2f}",
                                "Direction": opt["direction"].capitalize(),
                                "Change (%)": f"{opt['change_percent']:.2f}%",
                                "New Value": f"{opt['new_value']:.2f}"
                            })
                        else:  # categorical
                            categorical_optimizations.append({
                                "Variable": var,
                                "Recommended Category": opt["recommended_category"]
                            })
                    
                    st.write("### Numeric Variable Optimizations")
                    st.table(pd.DataFrame(numeric_optimizations))
                    
                    st.write("### Categorical Variable Recommendations")
                    st.table(pd.DataFrame(categorical_optimizations))

                    # Visualization of numeric optimization suggestions
                    fig = go.Figure()
                    for opt in numeric_optimizations:
                        fig.add_trace(go.Bar(
                            x=[opt["Variable"]],
                            y=[float(opt["Change (%)"].rstrip('%'))],
                            name=opt["Variable"],
                            text=[opt["Change (%)"]],
                            textposition='auto',
                        ))
                    
                    fig.update_layout(
                        title=f"Suggested Changes to Optimize {target_variable} (Numeric Variables)",
                        xaxis_title="Variables",
                        yaxis_title="Suggested Change (%)",
                        barmode='group'
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    st.write("### Interpretation of Optimization Results")
                    st.write("""
                    The optimization analysis uses Monte Carlo simulation to suggest changes to various input variables that could help achieve the target increase in the selected KPI. 
                    For numeric variables, positive percentages indicate an increase, while negative percentages indicate a decrease. 
                    For categorical variables, the recommended category is shown.
                    These suggestions are based on simulated scenarios and should be interpreted in the context of real-world feasibility and business constraints.
                    The simulation model considers complex interactions between variables, including both numeric and categorical ones, providing more comprehensive recommendations.
                    Datetime-related variables have been excluded from the optimization process and visualization.
                    Only categorical variables with 20 or fewer unique values have been included in the analysis.
                    """)

                    # Display validation results
                    st.write("### Validation Results")
                    st.write(f"Mean Absolute Error: {validation_results['MAE']:.4f}")
                    st.write(f"Root Mean Squared Error: {validation_results['RMSE']:.4f}")
                    st.write(f"R-squared: {validation_results['R2']:.4f}")
                    st.write(f"Expected Change: {validation_results['Expected Change']:.4f}")
                    st.write(f"Actual Change: {validation_results['Actual Change']:.4f}")
                    
                    # Interpret validation results
                    st.write("### Interpretation of Validation Results")
                    mae_ratio = validation_results['MAE'] / abs(validation_results['Expected Change'])
                    if mae_ratio < 0.1:
                        validity_message = "The simulation results are highly reliable."
                    elif mae_ratio < 0.3:
                        validity_message = "The simulation results are moderately reliable."
                    else:
                        validity_message = "The simulation results should be interpreted with caution."
                    
                    st.write(f"""
                    {validity_message}
                    
                    - Mean Absolute Error (MAE) represents the average difference between the expected and actual change.
                    - Root Mean Squared Error (RMSE) gives more weight to larger errors, providing insight into the simulation's worst-case performance.
                    - R-squared (R2) indicates how well the simulation explains the variance in the target variable.
                    
                    A lower MAE and RMSE, and a higher R2 suggest more reliable simulation results. The closer the Actual Change is to the Expected Change, the more accurate the simulation.
                    """)

        else:
            st.warning("No suitable columns found in the dataset to use as KPIs.")

else:
    st.info("Please upload a CSV or Excel file to begin the analysis.")

# Footer
st.markdown("<div class='footer'>© 2024 Comprehensive Retail Analytics Suite. All rights reserved.</div>", unsafe_allow_html=True)