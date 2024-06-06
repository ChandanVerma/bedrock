import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
import time

# Database setup
DATABASE_URL = "sqlite:///chats.db"
engine = create_engine(DATABASE_URL)

# Set page config
st.set_page_config(
    page_title="LLM Chat Tracker",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load chat logs from the database
@st.cache_data(ttl=5)
def load_data():
    query = "SELECT * FROM chat_logs"
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

data = load_data()
data['tokens_used'] = pd.to_numeric(data['tokens_used'], errors='coerce')
# Add title and description
st.title("LLM Chat Tracker")
st.markdown("This application tracks all the chats made to the Langchain Bedrock API along with token usage and session details.")
avg_tokens_per_request = 0
# Add key metrics
col1, col2, col3, col4, col5, col6 = st.columns(6)
total_requests = len(data)
avg_request_duration = data['timestamp'].diff().mean()
total_tokens = data['tokens_used'].sum()
avg_tokens_per_request = data['tokens_used'].mean()
total_costs = total_tokens * 0.0001
avg_cost_per_request = total_costs / total_requests if total_requests else 0
col1.metric("Total Requests", total_requests)
col2.metric("Avg Request Duration", f"{avg_request_duration.total_seconds()}s" if avg_request_duration else "0s")
try:
    col3.metric("Avg Tokens per Request", int(avg_tokens_per_request))
except: 
    col3.metric("Avg Tokens per Request", 0)
col4.metric("Total Costs", f"${total_costs:.2f}")
col5.metric("Avg Cost per Request", f"${avg_cost_per_request:.2f}")

# Add session details table
session_details = data[['session_id', 'user_feedback', 'modification_needed', 'history', 'model_response', 'timestamp', 'latency', 'tokens_used', 'cost']].copy()
session_details['start_time'] = session_details['timestamp'] - pd.to_timedelta(session_details['latency'], unit='s')
session_details['first_token'] = session_details.groupby('session_id')['timestamp'].transform('min')
session_details['cost'] = session_details['cost'] 
session_details['session_id'] = session_details['session_id'].apply(lambda x: f'<a href="#" onclick="showDetails(\'{x}\')">{x}</a>')
session_details['user_feedback'] = session_details['user_feedback'].str.split('\n').str[0]
session_details['model_response'] = session_details['model_response'].str.split('\n').str[0]
session_details = session_details.drop_duplicates(subset=['session_id', 'user_feedback', 'model_response']).reset_index(drop=True)
session_details = session_details.head(10)
st.markdown("## Session Details")
st.write(session_details.to_html(escape=False, index=False), unsafe_allow_html=True)

# Add charts
st.markdown("## Requests Per Time")
requests_per_time = data.set_index('timestamp').resample('H').count()
st.line_chart(requests_per_time['session_id'])
st.markdown("## Token Usage")
token_usage = data.groupby('session_id')['tokens_used'].sum().reset_index()
st.bar_chart(token_usage, x='session_id', y='tokens_used')

# Add JavaScript code to show the details of a particular record when the session ID is clicked
st.markdown(
    """
    <script>
    function showDetails(sessionId) {
        var details = document.getElementById(sessionId + '-details');
        details.style.display = 'block';
        var sessionIdInput = document.getElementById('session-id-input');
        sessionIdInput.value = sessionId;
        sessionIdInput.dispatchEvent(new Event('input', { bubbles: true }));
    }
    </script>
    """,
    unsafe_allow_html=True,
)

# Add a right sidebar to display the details of a particular record
with st.sidebar:
    session_id = st.text_input("Session ID", key="session-id-input")
    if session_id:
        # record = data[data['session_id'] == session_id].iloc[0]
        filtered_data = data[data['session_id'] == session_id]
        record = filtered_data.sort_values(by='timestamp', ascending=False).iloc[0]
        # record = data.loc[data.groupby('session_id')['timestamp'].idxmax()]
        st.write(f"## Session ID: {session_id}")
        # st.write(f"# History: #\n{record['history']}")
        st.text_area(label= f"# Client Feedback: #",  value= record['user_feedback'], height=200)
        st.text_area(label= f"# History: #",  value= record['history'], height=225)
        
        st.text_area(label= f"# Manual modification: #",  value= record['modification_needed'], height=100)
        st.text_area(label= f"# Model Response: #",  value= record['model_response'], height=150)
        # st.write(f"# User Feedback: #\n{record['user_feedback']}")
        
        # st.write(f"# Model Response: #\n{record['model_response']}")
        st.write(f"# Start Time:  {record['timestamp']}")
        st.write(f"# Latency:  {round(record['latency'], 2)} seconds")
        st.write(f"# Tokens Used:  {record['tokens_used']}")
        st.write(f"# Cost: ${record['cost']:.2f}")

# Rerun the script every 5 seconds to update the session details table
while True:
    time.sleep(5)
    st.experimental_rerun()
