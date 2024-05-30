import streamlit as st
import requests
import time
import json
# Set the URL of the Flask app
FLASK_APP_URL = "http://localhost:5000"

# Set the session state for the conversation chain
if 'conversation_chain' not in st.session_state:
    st.session_state.conversation_chain = []

# Define the Streamlit app layout
st.title("Chatbot")

# Get the user input
user_input = st.text_input("User input")
survey_id = st.text_input("survey id")

# Generate a response using the Flask app
if st.button("Write with AI"):
    # Generate a response using the Flask app
# if user_input:
    response = requests.post(
        f"{FLASK_APP_URL}/write_with_ai",
        json={
            "survey_id": survey_id,
            "comment": user_input
        },
        stream=True
    )
    feedback = ""
    with st.chat_message("assistant"):
        placeholder = st.empty()
        for line in response.iter_lines():
            if line:
                time.sleep(0.2)
                feedback += line.decode("utf-8")
                placeholder.markdown(feedback)
    original = json.loads(response.headers['X-Original'])['original']
    secret_key = json.loads(response.headers['X-Secret-Key'])['secret_key']
    st.text_area(label="session_id", value=secret_key)
    st.session_state.conversation_chain.append(f"User: {user_input}")
    st.session_state.conversation_chain.append(f"AI: {feedback}")


# Revise the response using the Flask app
if 'conversation_chain' in st.session_state and st.session_state.conversation_chain:
    original_response = st.session_state.conversation_chain[-1].split(": ")[1]
    # st.text_input(label="LLM GENERATE FEEDBACK", value=original_response)
    modifications = st.text_input(label="What would you like to revise in the LLM-generated feedback?")
    session_key = st.text_input(label="session key")
    if st.button("Revise"):
        # response = requests.post(
        #     f"{FLASK_APP_URL}/revise",
        #     json={
        #         "survey_id": survey_id,
        #         "original_comment": user_input,
        #         "modifications": modifications,
        #         "secret_key": session_key
        #     }
        # ).json()
        # st.session_state.conversation_chain.append(f"User: {modifications}")
        # st.session_state.conversation_chain.append(f"AI: {response}")
        # st.text_area(label= 'modified response', value=response["modified_response"])
        response = requests.post(
            f"{FLASK_APP_URL}/revise",
            json={
                    "survey_id": survey_id,
                    "original_comment": user_input,
                    "modifications": modifications,
                    "secret_key": session_key
                },
            stream=True
        )
        feedback = ""
        with st.chat_message("assistant"):
            placeholder = st.empty()
            for line in response.iter_lines():
                if line:
                    time.sleep(0.5)
                    feedback += line.decode("utf-8")
                    placeholder.markdown(feedback)
    # original = json.loads(response.headers['X-Original'])['original']
    # secret_key = json.loads(response.headers['X-Secret-Key'])['secret_key']
    # st.text_area(label="session_id", value=secret_key)
    st.session_state.conversation_chain.append(f"User: {user_input}")
    # st.session_state.conversation_chain.append(f"AI: {feedback}")
# Display the conversation history
# if 'conversation_chain' in st.session_state and st.session_state.conversation_chain:
#     st.text_area("Conversation history", "\n".join(st.session_state.conversation_chain))
