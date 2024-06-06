from flask import Flask, request, jsonify, session, Response
import boto3
import secrets
import json
import os
from botocore.config import Config
from langchain_aws import ChatBedrock
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
import pandas as pd
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from count_tokens.count import count_tokens_in_string
import time
import json
load_dotenv()


# messages = [
#     HumanMessage(content='The username is Joseph Brown who provided the feedback. Respond to his feedback'),
#     AIMessage(content=" Dear Joseph Brown,\n\nThank you for taking the time to share your experience with our graphics card. We are truly sorry to hear that you have been encountering overheating issues, resulting in reduced system performance. We understand how frustrating this can be, especially when the product is expected to handle heavy loads.\n\nYour feedback regarding the cooling system not meeting the demands placed on the card is valuable to us. We take these concerns seriously and will investigate the matter further. Our team of engineers will analyze the issue and work on potential solutions to improve the card's thermal management.\n\nIn the meantime, we would suggest checking the following:\n\n1. Ensure proper airflow around the graphics card by checking the case fans and making sure they are functioning correctly.\n2. Make sure the graphics card is seated correctly in its slot and that all power connectors are securely attached.\n3. Update the graphics card driver to the latest version.\n\nIf the issue persists after trying these steps, please contact our technical support team for further assistance. They will be able to provide more detailed troubleshooting steps and potentially offer a replacement if necessary.\n\nOnce again, we apologize for any inconvenience caused and appreciate your patience as we work on a resolution. Your feedback is essential in helping us improve our products and provide the best possible experience for our customers.\n\nBest regards,\n[Your Name]\nEcommerce Customer Support Team.")
# ]

app_secret_key = secrets.token_hex(16)
print(app_secret_key)
app = Flask(__name__)
app.secret_key = app_secret_key

# Database setup
DATABASE_URL = "sqlite:///chats.db"
# Check if the database file exists
if os.path.exists(DATABASE_URL):
    # If the database file exists, delete it
    os.remove(DATABASE_URL)
    print("Existing database deleted.")

Base = declarative_base()

class ChatLog(Base):
    __tablename__ = 'chat_logs'
    id = Column(Integer, primary_key=True)
    session_id = Column(String(32))
    user_feedback = Column(String)
    modification_needed = Column(String)
    history = Column(String)
    model_response = Column(String)
    timestamp = Column(DateTime, default=datetime.now())
    latency = Column(Integer)
    tokens_used = Column(Integer)
    cost = Column(Integer)

engine = create_engine(DATABASE_URL)
Base.metadata.create_all(engine)  # This creates the table if it doesn't exist
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
db = SessionLocal()

retry_config = Config(
    region_name=os.environ.get("region_name"),
    retries={
        'max_attempts': 10,
        'mode': 'standard'
    }
)

store = {}
def get_session_history(session_id: str, number_of_messages_to_save_as_history=3) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    history = store[session_id]
    if len(history.messages) > number_of_messages_to_save_as_history:
        history.messages = history.messages[-number_of_messages_to_save_as_history:]
    return history

session = boto3.Session()
boto3_bedrock_runtime = session.client(service_name='bedrock-runtime', 
                                       aws_access_key_id=os.environ.get("aws_access_key_id"),
                                       aws_secret_access_key=os.environ.get("aws_secret_access_key"),
                                       config=retry_config)

model = ChatBedrock(
    model_id=os.getenv("BEDROCK_MODEL"),
    client=boto3_bedrock_runtime,
    verbose=True,
    streaming=True,
    model_kwargs={
        "temperature": 0.5,
        "top_p": 0.9,
    },
)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """I want you to act as a ecommerce customer support feedback replier.
               In a polite tone, respond to the product review given below: {review}. Respond to the best of your ability""",
        ),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ]
)

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

def get_username(user_id):
    df = pd.read_excel(os.getenv("data_path"))
    user_row = df[df['Survey ID'] == user_id]
    
    if not user_row.empty:
        return user_row.iloc[0]['Contact']
    return None

def log_chat(session_id, user_feedback, modification_needed, history, model_response, timestamp, latency, tokens_used, cost):
    chat_log = ChatLog(
        session_id=session_id,
        user_feedback=user_feedback,
        modification_needed = modification_needed, 
        history = history,
        model_response=model_response,
        timestamp=timestamp,
        latency=latency,
        tokens_used=tokens_used,
        cost = cost,
    )
    db.add(chat_log)
    db.commit()


def calculate_prompt_cost(prompt, model_name = os.getenv("BEDROCK_MODEL")):
    # Load pricing information from pricing.json
    with open('pricing.json', 'r') as file:
        pricing_data = json.load(file)
    
    # Check if the model name exists in the pricing data
    if model_name in pricing_data["chat"]:
        # Extract prompt price for the given model name
        prompt_price = pricing_data["chat"][model_name]["promptPrice"]
        
        # Calculate prompt cost
        prompt_cost = len(prompt.split()) * prompt_price
        return prompt_cost
    else:
        return None

def calculate_completion_cost(prompt, model_name = os.getenv("BEDROCK_MODEL")):
    # Load pricing information from pricing.json
    with open('pricing.json', 'r') as file:
        pricing_data = json.load(file)
    
    # Check if the model name exists in the pricing data
    if model_name in pricing_data["chat"]:
        # Extract completion price for the given model name
        completion_price = pricing_data["chat"][model_name]["completionPrice"]
        
        # Calculate completion cost
        completion_cost = len(prompt.split()) * completion_price
        return completion_cost
    else:
        return None


def generate_response(username, comment, secret_key=None):
    config = {"configurable": {"session_id": secret_key}}
    response = ""
    tokens_used = 0
    st = time.time()
    for chunk in write_with_ai_history.stream(
        {"review": f"""{comment}""",
         "input": f"""The username is {username} who provided the feedback. Respond to his feedback""",
         "history": ""},
        config=config,
    ):
        response += chunk
        tokens_used += count_tokens_in_string(chunk)
    et = time.time()
    latency = et - st
    cost = calculate_prompt_cost(prompt=comment, model_name=os.getenv("BEDROCK_MODEL")) + \
           calculate_completion_cost(prompt = response, model_name=os.getenv("BEDROCK_MODEL"))
    print(f"NUMBER OF TOKENS USED: {tokens_used}")
    log_chat(session_id=secret_key, 
             user_feedback=comment, 
             modification_needed = "",
             history="",
             model_response=response,
             timestamp=datetime.now(), 
             latency=latency, 
             tokens_used=tokens_used,
             cost = cost)
    return response, comment, secret_key

def modify_response(revision, secret_key=None):
    config = {"configurable": {"session_id": secret_key}}
    response = ""
    tokens_used = 0
    history = get_session_history(secret_key)
    # hist_str = str([{'content': message.content} for message in history])
    print(str(history))
    print(f"HISTORY IS: {type(history)}")
    st = time.time()
    for chunk in revise_comment_history.stream(
        {
        "history": history,
        "review": """""",
        "input": f"{revision}"
    },
        config=config,
    ):
        response += chunk
        tokens_used += len(chunk.split())
    et = time.time()
    latency = et - st
    cost = calculate_prompt_cost(prompt=revision, model_name=os.getenv("BEDROCK_MODEL")) + \
           calculate_completion_cost(prompt = response, model_name=os.getenv("BEDROCK_MODEL"))
    print(f"NUMBER OF TOKENS USED: {tokens_used}")
    log_chat(session_id=secret_key, 
             user_feedback="",
             modification_needed = revision, 
             history=str(history), 
             model_response=response,
             timestamp=datetime.now(), 
             latency=latency, 
             tokens_used=tokens_used,
             cost = cost)
    return response

@app.route('/write_with_ai', methods=['POST'])
def write_with_ai():
    secret_key = secrets.token_hex(16)
    
    user_id = int(request.json['survey_id'])
    comment = request.json['comment']

    username = get_username(user_id)
    feedback, original, secret_key = generate_response(username=username, 
                                 comment=comment, 
                                 secret_key=secret_key)

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

    username = get_username(survey_id)
    modified_response = modify_response(
        modifications,
        secret_key
    )

    return Response(modified_response, mimetype='text/event-stream')

if __name__ == '__main__':
    app.run(debug=True)
