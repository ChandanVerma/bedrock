from flask import Flask, request, jsonify, session, Response
import boto3
import csv
from langchain.chains import ConversationChain
import secrets
import csv
import json
from pydantic import BaseModel, Field
import os
from botocore.config import Config
from langchain_aws import ChatBedrock, Bedrock
from langchain_core.messages import HumanMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain.output_parsers import PydanticOutputParser
from langchain_core.output_parsers import StrOutputParser


app_secret_key = secrets.token_hex(16)
print(app_secret_key)
app = Flask(__name__)
app.secret_key = app_secret_key

# Initialize AWS SDK

# Configuring Boto3
retry_config = Config(
    region_name='ap-south-1',
    retries={
        'max_attempts': 10,
        'mode': 'standard'
    }
)

store = {}
def get_session_history(session_id: str, number_of_messages_to_save_as_history = 3) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    history = store[session_id]
    # Trim the history to keep only the last 3 messages
    if len(history.messages) > number_of_messages_to_save_as_history:
        history.messages = history.messages[-number_of_messages_to_save_as_history:]
    return history

session = boto3.Session()
boto3_bedrock_runtime = session.client(service_name='bedrock-runtime', 
                                       aws_access_key_id = os.environ.get("aws_access_key_id"),
                                       aws_secret_access_key = os.environ.get("aws_secret_access_key"),
                                       config=retry_config) #creates a Bedrock client

s3_client = session.client(service_name = 's3', 
                           aws_access_key_id = os.environ.get("aws_access_key_id"),
                           aws_secret_access_key = os.environ.get("aws_secret_access_key"),
                           config=retry_config)



model = ChatBedrock(
    model_id="mistral.mistral-7b-instruct-v0:2",
    client=boto3_bedrock_runtime,
    verbose=True,
    streaming=True,
    model_kwargs={
        # "max_tokens_to_sample": 100,
        "temperature": 0.5,
        "top_p": 0.9,
        # "generation_token_count": 100
    },
)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """I want you to act as a ecommerce customer support feeeback replier.
               In a polite tone, respond to the product review given below: {review}. Respond to the best of your ability""",
        ),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ]
)

# class ResponseOutput(BaseModel):
#     feedback_response: str = Field(description="The response from the LLM after reading the user feefack")

# # output = PydanticOutputParser(pydantic_object=ResponseOutput)
output = StrOutputParser()

runnable = prompt | model | output

revise_comment_history = RunnableWithMessageHistory(
    runnable,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

write_with_ai_history = RunnableWithMessageHistory(
    runnable,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

# for chunk in write_with_ai_history.stream(
#     {"review": """Poor Performance: Despite the specifications promising high performance, I experienced significant lag and stuttering even in moderate gaming and graphic-intensive applications. The frame rates were inconsistent, and it did not meet my expectations for an 8GB graphics card.""",
#     "input": """Respond to this feedback""",
#     "history": ""},
#     config={"configurable": {"session_id": app_secret_key}},
# ):
#     print(chunk, end="", flush=True)




import pandas as pd

# Function to search for the username in the user_info.csv or database
def get_username(user_id):
    # Read the Excel file
    df = pd.read_excel('/home/chandanv/Projects/bedrock/user_data.xlsx')
    
    # Find the row that matches the user_id
    user_row = df[df['Survey ID'] == user_id]
    
    if not user_row.empty:
        return user_row.iloc[0]['Contact']
    return None

# name = get_username(7885020)

# # Function to generate a response using Amazon Bedrock
# def generate_response(username, comment, secret_key = None):
#     # Add the user's comment to the conversation chain
#     config = {"configurable": {"session_id": secret_key}}
#     response = write_with_ai_history.invoke(
#                                 {"review": f"""{comment}""", 
#                                 "input": f"""The username is {username} who provided the feedback. Respond to his feedback""",
#                                 "history": ""},
#                                 config=config,
#                         )
#     # Generate a response using Amazon Bedrock
#     return response, comment, secret_key

def generate_response(username, comment, secret_key=None):
    config = {"configurable": {"session_id": secret_key}}
    response = ""
    for chunk in write_with_ai_history.stream(
        {"review": f"""{comment}""",
         "input": f"""The username is {username} who provided the feedback. Respond to his feedback""",
         "history": ""},
        config=config,
    ):
        response += chunk
    return response, comment, secret_key



def modify_response(revision, secret_key = None):
    config = {"configurable": {"session_id": secret_key}}
    response = ""
    for chunk in revise_comment_history.stream(
        {
        "history": get_session_history(secret_key),
        "review": """""",
        "input": f"{revision}"
    },
        config=config,
    ):
        response += chunk
    return response

# Function to modify the response using Amazon Bedrock
# def modify_response(revision, secret_key = None):
#     # Generate a modified response using Amazon Bedrock
#     config = {"configurable": {"session_id": secret_key}}
#     response_2 = revise_comment_history.invoke(
#     {
#         "history": get_session_history(secret_key),
#         "review": """""",
#         "input": f"{revision}"
#     },
#     config=config,
#     )

#     return response_2




# Function to store a file in an S3 bucket
def store_file_in_s3(file_content, bucket_name, file_key):
    s3_client.put_object(
        Body=file_content,
        Bucket=bucket_name,
        Key=file_key
    )

@app.route('/write_with_ai', methods=['POST'])
def write_with_ai():
    secret_key = secrets.token_hex(16)
    
    user_id = int(request.json['survey_id'])
    comment = request.json['comment']

    # Search for the username
    username = get_username(user_id)

    # Initialize the conversation chain for the user if it doesn't exist
    # Generate a response using Amazon Bedrock
    # response = generate_response(username, comment, session['conversation_chain'])
    feedback, original, secret_key = generate_response(username=username, 
                                 comment=comment, 
                                 secret_key=secret_key)

    # Store the response in an S3 bucket
    # store_file_in_s3(response, 'your-bucket-name', f'responses/{user_id}.txt')

    # Return the response as JSON or a string of text
    # return jsonify({'response': feeback, 
    #                 "original_feedback": original,
    #                 "secret_key": secret_key})
    headers = {
        'X-Original': json.dumps({'original': original}),
        'X-Secret-Key': json.dumps({'secret_key': secret_key})
    }
    return Response(feedback, mimetype='text/event-stream', headers=headers)

@app.route('/revise', methods=['POST'])
def revise():
    survey_id = request.json['survey_id']
    original_comment = request.json['original_comment']
    modifications = request.json['modifications']
    secret_key = request.json['secret_key']

    # Search for the username
    username = get_username(survey_id)

    # Modify the response using Amazon Bedrock
    modified_response = modify_response( 
                                        modifications,
                                        secret_key)

    # Store the modified response in an S3 bucket
    # store_file_in_s3(modified_response, 'your-bucket-name', f'modified_responses/{user_id}.txt')

    # Return the modified response as JSON or a string of text
    # return jsonify({'modified_response': modified_response})
    return Response(modified_response, mimetype='text/event-stream')

if __name__ == '__main__':
    app.run(debug=True)



import requests

response = requests.post(
    'http://localhost:5000/write_with_ai',
    json={
        'survey_id': 123,
        'comment': 'This is a test comment.'
    },
    stream=True
)
import time
for line in response.iter_lines():
    if line:
        print(line.decode('utf-8'))
        time.sleep(0.5)
