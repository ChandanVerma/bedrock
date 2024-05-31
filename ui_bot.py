import streamlit as st
import requests

# Set the URL of the Flask app
FLASK_APP_URL = "http://localhost:5000"

# Set the session state for the conversation chain
if 'conversation_chain' not in st.session_state:
    st.session_state.conversation_chain = []

# Define the Streamlit app layout
st.title("Chatbot")

# Get the user input
user_input = st.text_input("User input")

# Generate a response using the Flask app
if st.button("Write with AI"):
    response = requests.post(
        f"{FLASK_APP_URL}/write_with_ai",
        json={
            "user_id": "your-user-id",
            "comment": user_input
        },
        stream=True
    )
    output = ""
    for line in response.iter_lines():
        if line:
            output += line.decode("utf-8")
            st.text_area("Response", output)
    st.session_state.conversation_chain.append(f"User: {user_input}")
    st.session_state.conversation_chain.append(f"AI: {output}")

# Revise the response using the Flask app
if 'conversation_chain' in st.session_state and st.session_state.conversation_chain:
    original_response = st.session_state.conversation_chain[-1].split(": ")[1]
    modifications = st.text_input("What would you like to revise in the LLM-generated feedback?", value=original_response)
    if st.button("Generate revision"):
        response = requests.post(
            f"{FLASK_APP_URL}/revise",
            json={
                "user_id": "your-user-id",
                "original_comment": user_input,
                "modifications": modifications
            },
            stream=True
        )
        output = ""
        for line in response.iter_lines():
            if line:
                output += line.decode("utf-8")
                st.text_area("Revised response", output)
        st.session_state.conversation_chain.append(f"User: {modifications}")
        st.session_state.conversation_chain.append(f"AI: {output}")

# Display the conversation history
if 'conversation_chain' in st.session_state and st.session_state.conversation_chain:
    st.text_area("Conversation history", "\n".join(st.session_state.conversation_chain))
