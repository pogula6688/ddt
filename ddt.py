import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

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
    .title {
        text-align: center;
        font-size: 2.5em;
        color: var(--primary-color);
        margin-bottom: 20px;
    }
    .header {
        text-align: center;
        font-size: 1.5em;
        margin-bottom: 10px;
    }
    .footer {
        text-align: center;
        font-size: 0.9em;
        color: gray;
        margin-top: 30px;
    }
</style>
""", unsafe_allow_html=True)

# Title of the app
st.markdown("<div class='title'>Dynamic Digital Twin Application for Retail Optimization</div>", unsafe_allow_html=True)

# Professional summary of the app
st.write("""
This application leverages advanced analytics and predictive capabilities to empower retail businesses. 
By utilizing AI-driven insights, it enables users to visualize data dynamically, identify trends, and optimize performance metrics. 
With robust data processing and interactive visualizations, this tool facilitates informed decision-making and strategic planning in the retail sector.
Upload your datasets to uncover valuable insights and enhance your predictive analytics capabilities.
""")

# File uploader to upload any dataset
uploaded_file = st.file_uploader("Upload your dataset (CSV, XLS, XLSX)", type=['csv', 'xls', 'xlsx'])

def preprocess_data(df):
    """Fill missing values with the mean for numerical columns, mode for categorical and datetime columns."""
    for column in df.columns:
        if df[column].dtype == 'object':
            df[column].fillna(df[column].mode()[0], inplace=True)
        elif df[column].dtype == 'datetime64[ns]':
            df[column].fillna(df[column].mode()[0], inplace=True)
        else:
            df[column].fillna(df[column].mean(), inplace=True)
    return df

def aggregate_data(df, group_col, agg_col, agg_func):
    """Aggregate the data based on the specified function."""
    if agg_func == 'Average':
        return df.groupby(group_col)[agg_col].mean().reset_index()
    elif agg_func == 'Total':
        return df.groupby(group_col)[agg_col].sum().reset_index()
    elif agg_func == 'Count':
        return df.groupby(group_col).size().reset_index(name='Count')
    return df

def format_x_axis(fig, x_col, df):
    """Dynamically format the x-axis based on data type."""
    if df[x_col].dtype in ['int64', 'float64']:
        if df[x_col].between(1900, 2100).all():
            fig.update_xaxes(tickvals=df[x_col].unique(), ticktext=[str(int(year)) for year in df[x_col].unique()])
    elif df[x_col].dtype == 'datetime64[ns]':
        fig.update_xaxes(tickformat='%Y-%m-%d')
    return fig

def convert_datetime(df):
    """Convert columns with datetime data."""
    for column in df.columns:
        if df[column].dtype == 'object':
            try:
                df[column] = pd.to_datetime(df[column])
            except (ValueError, TypeError):
                continue
    return df

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
    """Create a bar chart."""
    fig = px.bar(df, x=x_col, y=y_col, title=f"Bar Chart of {y_col} by {x_col}", hover_data=[x_col, y_col])
    fig.update_layout(title_font=dict(size=20), xaxis_title_font=dict(size=14), yaxis_title_font=dict(size=14))
    return fig

def create_line_chart(df, x_col, y_col):
    """Create a line chart."""
    fig = px.line(df, x=x_col, y=y_col, title=f"Line Chart of {y_col} by {x_col}", hover_data=[x_col, y_col])
    fig.update_layout(title_font=dict(size=20), xaxis_title_font=dict(size=14), yaxis_title_font=dict(size=14))
    return fig

# Initialize the generated_charts list
generated_charts = []

if uploaded_file is not None:
    # Load the dataset
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    # Display the dataframe
    st.dataframe(df)

    # Convert datetime columns
    df = convert_datetime(df)

    # Preprocess the data
    df = preprocess_data(df)

    # Check if the DataFrame is empty after preprocessing
    if df.empty:
        st.warning("The dataset is empty after preprocessing.")
    else:
        # Include all column types in the selection
        columns = df.columns.tolist()

        # Input to specify the number of charts
        num_charts = st.sidebar.number_input("How many charts do you want to generate?", min_value=1, max_value=10, value=1)

        for i in range(num_charts):
            st.sidebar.subheader(f"Chart {i + 1} Configuration")
            
            x_col = st.sidebar.selectbox(f"Select X-axis column for Chart {i + 1}:", options=[""] + columns, key=f"x_col_{i}")
            y_col = st.sidebar.selectbox(f"Select Y-axis column for Chart {i + 1}:", options=[""] + columns, key=f"y_col_{i}")
            agg_func = st.sidebar.selectbox(f"Select aggregation function for Chart {i + 1}:", [""] + ["Average", "Total", "Count"], key=f"agg_func_{i}")

            # Only aggregate if necessary and x_col and y_col are valid
            if x_col and agg_func:
                aggregated_df = aggregate_data(df, x_col, y_col, agg_func)
            else:
                aggregated_df = df

            chart_type = st.sidebar.selectbox(f"Select Chart Type for Chart {i + 1}:", [""] + ["Bar", "Line", "Doughnut"], key=f"chart_type_{i}")

            try:
                if x_col and y_col:
                    if chart_type == 'Doughnut':
                        threshold = 20  # Define the maximum number of unique categories allowed
                        fig = create_doughnut_chart(df, x_col, y_col, threshold)
                        generated_charts.append(fig)
                    elif chart_type == 'Bar':
                        if x_col in df.select_dtypes(include=['object']).columns and y_col in df.select_dtypes(include=['number']).columns:
                            fig = create_bar_chart(aggregated_df, x_col, y_col)
                            generated_charts.append(fig)
                        else:
                            raise ValueError("Bar chart requires a categorical x-axis and a numeric y-axis.")
                    elif chart_type == 'Line':
                        if (x_col in df.select_dtypes(include=['number', 'datetime64[ns]']).columns) and \
                           (y_col in df.select_dtypes(include=['number']).columns):
                            fig = create_line_chart(aggregated_df, x_col, y_col)
                            fig = format_x_axis(fig, x_col, aggregated_df)
                            generated_charts.append(fig)
                        else:
                            raise ValueError("Line chart requires a numeric or datetime x-axis and a numeric y-axis.")
            except ValueError as e:
                st.error(f"Input Error: {str(e)}")
            except Exception as e:
                st.error(f"An unexpected error occurred: {str(e)}")

# Display all generated charts in the specified layout
if generated_charts:
    st.subheader("Generated Dashboard")

    num_charts = len(generated_charts)
    if num_charts == 4:
        cols = st.columns(3)
        for i in range(3):
            with cols[i]:
                st.plotly_chart(generated_charts[i], use_container_width=True)
        st.plotly_chart(generated_charts[3], use_container_width=True, width=800)
    else:
        rows = (num_charts - 1) // 3 + 1
        for row in range(rows):
            cols = st.columns(3)
            for col in range(3):
                idx = row * 3 + col
                if idx < num_charts:
                    with cols[col]:
                        st.plotly_chart(generated_charts[idx], use_container_width=True)

# Footer
st.markdown("""
<div class='footer'>
    <p>Powered by Streamlit | Data Visualization and Analytics Tool</p>
    <p>For additional information, contact at: <a href='mailto:Pogula-V@ulster.ac.uk'>Pogula-V@ulster.ac.uk</a></p>
</div>
""", unsafe_allow_html=True)

# Display an info message if no charts are generated
if not generated_charts:
    st.info("No charts to display. Please configure and generate charts.")
