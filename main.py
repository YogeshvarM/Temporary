import streamlit as st
import pandas as pd
from langchain.agents import AgentType
from langchain_groq import ChatGroq
from langchain.agents import create_sql_agent
from langchain_community.utilities import SQLDatabase
from sqlalchemy import create_engine
import json
import os
import numpy as np
from typing import Dict, Any, List, Optional, Union
import sqlite3
import tempfile
from datetime import datetime
import base64
from pathlib import Path
import plotly.graph_objects as go
from streamlit.components.v1 import html
from langchain.agents import initialize_agent
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.tools import Tool
from langchain.prompts import PromptTemplate
import time
from tenacity import retry, stop_after_attempt, wait_exponential
import logging
from langchain.chains import LLMChain

# Add after imports, before page configuration
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Set page configuration
st.set_page_config(
    page_title="Data Analysis Assistant",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session states
def init_session_state():
    """Initialize all session state variables"""
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'conversation_memory' not in st.session_state:
        st.session_state.conversation_memory = ConversationBufferMemory(
            memory_key="history",
            input_key="input",
            return_messages=True
        )
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'db_path' not in st.session_state:
        st.session_state.db_path = None
    if 'data_overview' not in st.session_state:
        st.session_state.data_overview = None
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 'config'
    if 'is_configured' not in st.session_state:
        st.session_state.is_configured = False

def add_resizable_sidebar():
    """Add custom HTML/JavaScript for resizable sidebar"""
    sidebar_html = """
    <style>
        section[data-testid="stSidebar"] {
            position: relative;
            resize: horizontal;
            overflow: auto;
            min-width: 200px;
            max-width: 800px;
        }
        section[data-testid="stSidebar"]::after {
            content: '';
            position: absolute;
            right: 0;
            top: 0;
            bottom: 0;
            width: 5px;
            background: #e0e0e0;
            cursor: col-resize;
        }
    </style>
    """
    html(sidebar_html, height=0)

class DataAnalyzer:
    """Class to handle data analysis operations"""
    
    @staticmethod
    def get_data_overview(df: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive overview of the dataframe"""
        # Initialize the overview dictionary with all required keys
        overview = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'memory_usage': df.memory_usage(deep=True).sum() / 1024**2,  # In MB
            'columns': {},
            'correlations': None,
            'visualizations': {
                'metrics': [
                    {"label": "Rows", "value": len(df)},
                    {"label": "Columns", "value": len(df.columns)},
                    {"label": "Memory (MB)", "value": f"{df.memory_usage(deep=True).sum() / 1024**2:.1f}"}
                ],
                'distributions': {},
                'correlation_matrix': None
            }
        }
        
        # Calculate correlations for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            overview['correlations'] = df[numeric_cols].corr().to_dict()
            correlation_df = pd.DataFrame(overview['correlations'])
            overview['visualizations']['correlation_matrix'] = {
                'z': correlation_df.values.tolist(),
                'x': correlation_df.columns.tolist(),
                'y': correlation_df.columns.tolist()
            }
        
        for col in df.columns:
            col_info = {
                'dtype': str(df[col].dtype),
                'missing_values': df[col].isna().sum(),
                'missing_percentage': (df[col].isna().sum() / len(df)) * 100,
                'unique_values': df[col].nunique(),
                'memory_usage': df[col].memory_usage(deep=True) / 1024**2
            }
            
            # Add sample values and frequency for categorical/object columns
            if df[col].dtype == 'object' or df[col].dtype == 'category':
                col_info.update({
                    'sample_values': list(set(df[col].dropna().sample(min(5, df[col].nunique())).tolist())),
                    'top_values': df[col].value_counts().head(5).to_dict()
                })
            
            # Add statistical info for numeric columns
            if pd.api.types.is_numeric_dtype(df[col]):
                col_info.update({
                    'mean': df[col].mean() if not df[col].empty else None,
                    'median': df[col].median() if not df[col].empty else None,
                    'std': df[col].std() if not df[col].empty else None,
                    'min': df[col].min() if not df[col].empty else None,
                    'max': df[col].max() if not df[col].empty else None,
                    'quartiles': df[col].quantile([0.25, 0.5, 0.75]).to_dict()
                })
                
                # Add distribution data for numeric columns
                overview['visualizations']['distributions'][col] = {
                    'y': df[col].dropna().tolist(),
                    'name': col
                }
            
            # Detect potential datetime columns
            if df[col].dtype == 'object':
                try:
                    pd.to_datetime(df[col].dropna().head())
                    col_info['potential_datetime'] = True
                except:
                    col_info['potential_datetime'] = False
            
            overview['columns'][col] = col_info
        
        return overview

class DatabaseManager:
    """Class to handle database operations"""
    
    @staticmethod
    def setup_database(df: pd.DataFrame) -> str:
        """Create and setup SQLite database from dataframe"""
        # Create temporary file for SQLite database
        temp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        db_path = temp_db.name
        temp_db.close()
        
        # Create SQLite database and add data
        engine = create_engine(f'sqlite:///{db_path}')
        
        # Convert datetime columns to string to avoid SQLite limitations
        df_copy = df.copy()
        for col in df_copy.select_dtypes(include=['datetime64']):
            df_copy[col] = df_copy[col].astype(str)
        
        df_copy.to_sql('user_data', engine, index=False, if_exists='replace')
        
        return db_path

class APIKeyValidator:
    """Class to handle API key validation"""
    
    @staticmethod
    def validate_groq_api_key(api_key: str) -> bool:
        """Validate Groq API key by attempting to create a client"""
        try:
            llm = ChatGroq(
                temperature=0.1,
                groq_api_key=api_key,
                model_name="llama-3.3-70b-versatile",
                max_tokens=1
            )
            # Try a simple completion to verify the key
            llm.predict("test")
            return True
        except Exception:
            return False

class ChatInterface:
    """Class to handle chat interface and interactions"""
    
    @staticmethod
    def display_chat_history():
        """Display chat history with enhanced styling"""
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                if isinstance(message["content"], dict):
                    # Display text response
                    st.markdown(message["content"].get("text_response", ""))
                    # Display visualization if present
                    if "visualization" in message["content"] and message["content"]["visualization"] is not None:
                        st.plotly_chart(message["content"]["visualization"], use_container_width=True)
                else:
                    # Handle string content
                    st.markdown(str(message["content"]))

    @staticmethod
    def create_master_agent(llm, df):
        """Create a master agent to route queries with awareness of loaded data"""
        # Get column information for context
        columns_info = "\n".join([
            f"- {col} ({df[col].dtype})" for col in df.columns
        ])
        
        tools = [
            Tool(
                name="SQL_Database_Query",
                func=lambda x: "USE_SQL_AGENT",
                description=f"""Use this tool when:
                - User asks about the loaded dataset with columns:
                {columns_info}
                - Requests involve analyzing these columns
                - Questions about specific data fields or values in the dataset
                - Requests for trends, patterns, or insights from this data
                - Any query that needs to examine the actual data values
                Keep in mind we have a 7000 tokens per minute limit."""
            ),
            Tool(
                name="General_Conversation",
                func=lambda x: "USE_CONVERSATION",
                description=f"""Use this tool only when:
                - User is greeting without mentioning data
                - Questions about general capabilities
                - No specific reference to the loaded dataset columns:
                {columns_info}
                Keep responses concise due to 7000 tokens per minute limit."""
            )
        ]
        
        system_message = f"""You are a routing agent for a data analysis assistant working with a specific dataset.
        
        The loaded dataset contains the following columns:
        {columns_info}
        
        IMPORTANT: 
        1. We have a strict limit of 7000 tokens per minute.
        2. ALWAYS choose SQL_Database_Query when the user:
           - Mentions any of the available columns
           - Asks about data in the loaded dataset
           - Requests statistics or insights about this data
           - Uses terms that match or relate to our column names
        3. Only choose General_Conversation for:
           - Pure greetings with no data request
           - Questions about capabilities
           - General help without mentioning our data
        4. When in doubt, prefer SQL_Database_Query as we have data about: {', '.join(df.columns)}
        """
        
        return initialize_agent(
            tools,
            llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            handle_parsing_errors=True,
            system_message=system_message
        )

    @staticmethod
    def handle_conversation(llm, df, user_input: str) -> str:
        """Handle normal conversation with awareness of loaded data"""
        columns_info = "\n".join([
            f"- {col} ({df[col].dtype})" for col in df.columns
        ])
        
        prompt = PromptTemplate(
            input_variables=["history", "input"],
            template=f"""You are a Data Analysis Assistant working with a specific dataset.

            The current dataset contains these columns:
            {columns_info}

            IMPORTANT: 
            - Keep responses concise (7000 tokens/minute limit)
            - Always maintain awareness of available data columns
            - If users ask about capabilities, explain you can analyze this dataset
            - Refer to actual columns when discussing possibilities
            - If user asks about unavailable data, explain what we do have

            Current conversation:
            {{history}}
            Human: {{input}}
            Assistant:"""
        )

        conversation = ConversationChain(
            llm=llm,
            verbose=True,
            memory=ConversationBufferMemory(),
            prompt=prompt
        )
        return conversation.predict(input=user_input)
    @staticmethod
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def process_user_input(user_input: str, groq_api_key: str) -> Dict[str, Any]:
        """Process user input with improved flow control"""
        try:
            # Get conversation history from session state
            conversation_history = st.session_state.chat_history[-3:] if st.session_state.chat_history else []
            
            # Create context from previous messages
            context = "\n".join([
                f"{msg['role']}: {msg['content'].get('text_response', msg['content']) if isinstance(msg['content'], dict) else msg['content']}"
                for msg in conversation_history
            ])
            
            # Add context to user input
            contextualized_input = f"""Previous conversation:
{context}

Current request: {user_input}"""

            llm = ChatGroq(
                temperature=0.1,
                groq_api_key=groq_api_key,
                model_name="llama-3.1-70b-versatile",
                max_tokens=1000
            )
            
            df = st.session_state.df
            
            # Process the request
            if any(viz_word in user_input.lower() for viz_word in ['visual', 'plot', 'graph', 'chart', 'show']):
                # Direct visualization request
                viz_suggestion = ChatInterface.determine_visualization_type(contextualized_input, df, "")
                if viz_suggestion.get("visualize", False):
                    fig = ChatInterface.create_visualization(df, viz_suggestion["type"], viz_suggestion["columns"])
                    return {
                        "text_response": viz_suggestion["reason"],
                        "visualization": fig
                    }
            
            # Regular flow with SQL if needed
            master_agent = ChatInterface.create_master_agent(llm, df)
            route_decision = master_agent.invoke({"input": contextualized_input})
            
            if "USE_SQL_AGENT" in route_decision["output"]:
                db = SQLDatabase.from_uri(f"sqlite:///{st.session_state.db_path}")
                sql_agent = create_sql_agent(
                    llm=llm,
                    db=db,
                    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                    verbose=True
                )
                sql_response = sql_agent.invoke({"input": contextualized_input})
                
                response = {"text_response": sql_response["output"]}
                
                # Automatically determine if visualization would be helpful
                viz_suggestion = ChatInterface.determine_visualization_type(
                    contextualized_input,
                    df,
                    sql_response["output"]
                )
                
                if viz_suggestion.get("visualize", False):
                    fig = ChatInterface.create_visualization(
                        df,
                        viz_suggestion["type"],
                        viz_suggestion["columns"]
                    )
                    if fig:
                        response["visualization"] = fig
                
                return response
            else:
                # Handle regular conversation
                conversation_response = ChatInterface.handle_conversation(llm, df, contextualized_input)
                return {"text_response": conversation_response}
            
        except Exception as e:
            logger.error(f"Error in process_user_input: {str(e)}")
            return {"text_response": f"An error occurred: {str(e)}. Please try again."}

    @staticmethod
    def create_visualization(df: pd.DataFrame, viz_type: str, columns: Dict[str, str]) -> Optional[go.Figure]:
        """Create visualization based on type and column mapping"""
        try:
            fig = go.Figure()
            
            if viz_type == "grouped_bar":
                # Create grouped bar chart for multi-variable analysis
                if columns.get("color"):
                    # Calculate mean values for each group
                    grouped_data = df.groupby(columns["x"])[columns["y"]].mean().reset_index()
                    
                    # Add bars for each category
                    for category in df[columns["color"]].unique():
                        category_data = df[df[columns["color"]] == category]
                        means = category_data.groupby(columns["x"])[columns["y"]].mean()
                        
                        fig.add_trace(go.Bar(
                            name=f"{category}",
                            x=means.index,
                            y=means.values,
                            text=means.values.round(2),
                            textposition='auto',
                        ))
                    
                    fig.update_layout(
                        barmode='group',
                        title=f"{columns['y']} by {columns['x']} and {columns['color']}",
                        xaxis_title=columns["x"],
                        yaxis_title=columns["y"],
                        legend_title=columns["color"]
                    )
                else:
                    # Simple bar chart if no color grouping
                    grouped_data = df.groupby(columns["x"])[columns["y"]].mean().reset_index()
                    fig.add_trace(go.Bar(
                        x=grouped_data[columns["x"]],
                        y=grouped_data[columns["y"]],
                        text=grouped_data[columns["y"]].round(2),
                        textposition='auto',
                    ))
            
            elif viz_type == "scatter":
                fig.add_trace(go.Scatter(
                    x=df[columns["x"]],
                    y=df[columns["y"]],
                    mode='markers',
                    marker=dict(
                        size=8,
                        opacity=0.6,
                        line=dict(width=1)
                    )
                ))
                
            elif viz_type == "line":
                # Sort by x-axis for proper line connection
                sorted_df = df.sort_values(columns["x"])
                fig.add_trace(go.Scatter(
                    x=sorted_df[columns["x"]],
                    y=sorted_df[columns["y"]],
                    mode='lines+markers'
                ))
                
            elif viz_type == "box":
                fig.add_trace(go.Box(
                    y=df[columns["y"]],
                    name=columns["y"],
                    boxpoints='outliers'
                ))
                
            # Update layout with better styling
            fig.update_layout(
                title=f"{columns.get('y', '')} by {columns.get('x', '')}" if 'x' in columns else f"Distribution of {columns.get('y', '')}",
                xaxis_title=columns.get("x", ""),
                yaxis_title=columns.get("y", ""),
                height=500,
                template="plotly_white",
                showlegend=True,
                margin=dict(l=40, r=40, t=40, b=40)
            )
            
            return fig
        except Exception as e:
            logger.error(f"Visualization creation error: {str(e)}")
            return None

    @staticmethod
    def optimize_sql_query(query: str) -> str:
        """Optimize SQL queries for better performance"""
        # Add query optimization logic
        if "SELECT *" in query:
            # Replace SELECT * with specific columns
            return query.replace("SELECT *", "SELECT DISTINCT")
        
        # Add LIMIT for large result sets
        if "LIMIT" not in query.upper():
            return f"{query} LIMIT 1000"
        
        return query

    @staticmethod
    def determine_visualization_type(question: str, data: pd.DataFrame, result: str) -> Dict[str, Any]:
        """Determine appropriate visualization type based on question and data"""
        try:
            # Extract mentioned columns from the question
            columns = data.columns.tolist()
            mentioned_cols = [col for col in columns if col.lower() in question.lower()]
            
            # Multi-variable analysis (3 variables)
            if len(mentioned_cols) >= 3:
                numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns
                categorical_cols = data.select_dtypes(include=['object', 'category']).columns
                
                # For Age vs Gender vs Stress Level type analysis
                if any(col in categorical_cols for col in mentioned_cols):
                    cat_col = next(col for col in mentioned_cols if col in categorical_cols)
                    num_cols = [col for col in mentioned_cols if col in numeric_cols]
                    
                    if len(num_cols) >= 1:
                        return {
                            "visualize": True,
                            "type": "grouped_bar",  # New visualization type
                            "columns": {
                                "x": cat_col,
                                "y": num_cols[0],
                                "color": num_cols[1] if len(num_cols) > 1 else None
                            },
                            "reason": f"Comparing {num_cols[0]} across {cat_col} categories"
                        }
            
            # Existing logic for other cases...
            
        except Exception as e:
            logger.error(f"Visualization determination error: {str(e)}")
            return {"visualize": False}

def main():
    """Main application function"""
    init_session_state()
    
    if not st.session_state.is_configured:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.title("📊 Data Analysis Assistant")
            
            # Configuration page with improved validation
            groq_api_key = st.text_input(
                "Enter Groq API Key",
                type="password",
                help="Enter your Groq API key. Get one at https://console.groq.com"
            )
            
            if groq_api_key:
                with st.spinner("Validating API key..."):
                    if not APIKeyValidator.validate_groq_api_key(groq_api_key):
                        st.error("❌ Invalid API key. Please check your key and try again.")
                        return
            
            uploaded_file = st.file_uploader(
                "Upload your data file",
                type=['csv', 'xlsx', 'json', 'parquet'],
                help="Supported formats: CSV, Excel, JSON, Parquet"
            )
            
            if uploaded_file and groq_api_key:
                try:
                    with st.spinner("Loading and processing your data..."):
                        # File loading logic with improved error handling
                        file_extension = uploaded_file.name.split('.')[-1].lower()
                        
                        if file_extension not in ['csv', 'xlsx', 'json', 'parquet']:
                            st.error("❌ Unsupported file format")
                            return
                            
                        # Load data based on file type
                        df = None
                        try:
                            if file_extension == 'csv':
                                df = pd.read_csv(uploaded_file)
                            elif file_extension == 'xlsx':
                                df = pd.read_excel(uploaded_file)
                            elif file_extension == 'json':
                                df = pd.read_json(uploaded_file)
                            elif file_extension == 'parquet':
                                df = pd.read_parquet(uploaded_file)
                        except Exception as e:
                            st.error(f"❌ Error reading file: {str(e)}")
                            return
                        
                        if df is None or df.empty:
                            st.error("❌ The uploaded file contains no data")
                            return
                            
                        # Update session state
                        st.session_state.df = df
                        st.session_state.data_overview = DataAnalyzer.get_data_overview(df)
                        st.session_state.db_path = DatabaseManager.setup_database(df)
                        st.session_state.groq_api_key = groq_api_key
                        st.session_state.is_configured = True
                        
                        st.success("✅ Configuration successful! Redirecting to main interface...")
                        st.rerun()
                        
                except Exception as e:
                    st.error(f"❌ Error during configuration: {str(e)}")
    
    else:
        # Add resizable sidebar
        add_resizable_sidebar()
        
        # Sidebar content
        with st.sidebar:
            st.markdown("### 📊 Data Overview")
            
            # Display metrics
            if 'visualizations' in st.session_state.data_overview:
                cols = st.columns(3)
                for idx, metric in enumerate(st.session_state.data_overview['visualizations']['metrics']):
                    cols[idx].metric(metric["label"], metric["value"])
            
            # Column Analysis in sidebar
            st.markdown("#### Columns")
            for col_name, col_info in st.session_state.data_overview['columns'].items():
                with st.expander(f"{col_name}"):
                    st.write(f"📝 Type: {col_info['dtype']}")
                    st.write(f"❌ Missing: {col_info['missing_percentage']:.1f}%")
                    st.write(f"🔢 Unique: {col_info['unique_values']}")
                    
                    # Display sample unique values
                    sample_values = st.session_state.df[col_name].dropna().unique()[:5]  # Get first 5 unique values
                    if len(sample_values) > 0:
                        st.write("📋 Sample Values:")
                        for val in sample_values:
                            st.write(f"  • {val}")
                    
                    # Show distribution plot for numeric columns
                    if ('visualizations' in st.session_state.data_overview and 
                        'distributions' in st.session_state.data_overview['visualizations'] and
                        col_name in st.session_state.data_overview['visualizations']['distributions']):
                        try:
                            data = st.session_state.data_overview['visualizations']['distributions'][col_name]
                            fig = go.Figure()
                            fig.add_trace(go.Box(
                                y=data['y'],
                                name='Distribution'
                            ))
                            fig.update_layout(
                                title=f"Distribution of {col_name}",
                                showlegend=False,
                                height=300
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        except Exception as e:
                            st.error(f"Could not create visualization: {str(e)}")
            
            # Safely check for correlation matrix
            if ('visualizations' in st.session_state.data_overview and 
                'correlation_matrix' in st.session_state.data_overview['visualizations'] and
                st.session_state.data_overview['visualizations']['correlation_matrix']):
                st.subheader("Correlation Analysis")
                corr_data = st.session_state.data_overview['visualizations']['correlation_matrix']
                fig = go.Figure(data=go.Heatmap(
                    z=corr_data['z'],
                    x=corr_data['x'],
                    y=corr_data['y'],
                    colorscale='RdBu',
                    zmin=-1,
                    zmax=1
                ))
                fig.update_layout(
                    title='Correlation Matrix',
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Reset button
            st.divider()
            if st.button("⚠️ Reset", use_container_width=True):
                st.session_state.clear()
                st.rerun()
        
        # Main chat area - full width
        st.title("💬 Chat with Your Data")
        ChatInterface.display_chat_history()
        
        if user_input := st.chat_input("Ask me about your data"):
            st.session_state.chat_history.append(
                {"role": "user", "content": user_input}
            )
            
            with st.spinner("Thinking..."):
                response = ChatInterface.process_user_input(
                    user_input, 
                    st.session_state.groq_api_key
                )
                
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": response}
                )
                
                st.rerun()

if __name__ == "__main__":
    main()