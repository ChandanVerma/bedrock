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
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
import pandas as pd
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from count_tokens.count import count_tokens_in_string
import time
import json
from pydantic import BaseModel, Field

load_dotenv()


app_secret_key = secrets.token_hex(16)
print(app_secret_key)
app = Flask(__name__)
app.secret_key = app_secret_key

# Database setup
DATABASE_URL = "sqlite:///chats.db"
# Check if the database file exists

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

#####################################################

class ReviewsOutput(BaseModel):
    Response: str = Field(description="This will be the response that will be generate by you in reponse to the customer review.")
    Summary: str = Field(description="This will be the summary of the converstion based on the review provided.")
    Sentiment: str = Field(description="promoter or non-promoter")
    Positive_Themes: list = Field(description="This will be a python list of positives out of the entire conversation. Return a blank python list if there are no positives.")
    Negative_Themes: list = Field(description="This will be a python list of negatives out of the entire conversation. Return a blank python list if there are no negatives.")

output_parser = JsonOutputParser(pydantic_object=ReviewsOutput)

# prompt = ChatPromptTemplate.from_messages(
#     [
#         (
#             "system",
#             """
#             {review}.
#             Based on the Survey data provided generate a summary of the conversation, sentiment, positive_themes and negative themes
#             {survey_data}
#             format you response in a json format with the following keys:
#             Response: This will be the response that will be generate by you in reponse to the customer review.
#             Summary: This will be the summary of the converstion based on the {review} provided.
#             Sentiment: This will be based on the {survey_data} on the score column. promoter or non-promoter.
#             Positive_Themes: This will be a python list of positives out of the entire conversation. Return a blank python list if there are no positives.
#             Negative_Themes: This will be a python list of Negatives out of the entire conversation, Return a blank python list if there are no negatives.
#             Donot merge multiple keys responses in a single key.
            
#             Recheck the response is strictly in a json format with the following keys and nothings else:
#             Response, Summary, Sentiment, Positive Themes, Negative Themes""",
#         ),
#         MessagesPlaceholder(variable_name="history"),
#         ("human", "{input}"),
#     ]
# )


prompt = PromptTemplate(
    template="""
            {review}.
            Based on the Survey data provided generate a summary of the conversation, sentiment, positive_themes and negative themes
            {survey_data}
            format you response in a json format with the following keys:
            Response: This will be the response that will be generate by you in reponse to the customer review.
            Summary: This will be the summary of the converstion based on the {review} provided.
            Sentiment: This will be based on the {survey_data} on the score column. promoter or non-promoter.
            Positive_Themes: This will be a python list of positives out of the entire conversation. Return a blank python list if there are no positives.
            Negative_Themes: This will be a python list of Negatives out of the entire conversation, Return a blank python list if there are no negatives.
            Donot merge multiple keys responses in a single key.
            
            Recheck the response is strictly in a json format with the following keys and nothings else:
            Response, Summary, Sentiment, Positive Themes, Negative Themes""",
    input_variables=["review", "survey_data"],
    partial_variables={"format_instructions": output_parser.get_format_instructions()},
    

)

runnable = prompt | model | output_parser

write_with_ai_history = RunnableWithMessageHistory(
    runnable,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

############################


consolidated_summary_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            {review}.
            Summarize the response to capture the sentiments of the users in a maximum of two sentences.""",
        ),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ]
)

output = StrOutputParser()

consolidated_summary_runnable = consolidated_summary_prompt | model | output

consolidated_summary_history = RunnableWithMessageHistory(
    consolidated_summary_runnable,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

#################################################################

revise_comment_history = RunnableWithMessageHistory(
    runnable,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)


def get_username(json_data):
    try:
        return json_data.get("Feedback Data")["Customer"]["Name"]
    except:
        print("Customer Name does not exists")
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


def generate_response(username, 
                      user_id, 
                      context_prompt_0,
                      context_prompt_1,
                      core_prompt,
                      survey_data,
                      review, 
                      secret_key=None):
    config = {"configurable": {"session_id": secret_key}}
    response = ""
    tokens_used = 0
    st = time.time()
    response =  write_with_ai_history.invoke(
        {"review": f"""{review}""",
         "survey_data": f"""{survey_data}""",
         "input": f"""{context_prompt_1} 
                      {context_prompt_0}
                      Please adhere to the following policies:
                      {core_prompt} 
                      The username is {username} who provided the feedback.""",
         "history": ""},
        config=config,
    )
    # response = output_parser.parse(response)
    print(f"within generate response: {response}")
    # response =json.loads(response)
    print(type(response))
    response["ID"] = user_id
    # print(response)

    # tokens_used += count_tokens_in_string(response)
    # et = time.time()
    # latency = et - st
    # cost = calculate_prompt_cost(prompt=review, model_name=os.getenv("BEDROCK_MODEL")) + \
    #        calculate_completion_cost(prompt = response, model_name=os.getenv("BEDROCK_MODEL"))
    # print(f"NUMBER OF TOKENS USED: {tokens_used}")
    # log_chat(session_id=secret_key, 
    #          user_feedback=review, 
    #          modification_needed = "",
    #          history="",
    #          model_response=response,
    #          timestamp=datetime.now(), 
    #          latency=latency, 
    #          tokens_used=tokens_used,
    #          cost = cost)
    return response, review, secret_key


def generate_consolidated_summary_response(
                      review, 
                      secret_key=None):
    config = {"configurable": {"session_id": secret_key}}
    response = ""
    tokens_used = 0
    st = time.time()
    response =  consolidated_summary_history.invoke(
        {"review": f"""{review}""",
         "input": f"""Below are the reviews from {len(review)} user.
                      Summarize all the reviews into a single one to capture the sentiments of the all users in a maximum of 2 sentences.""",
         "history": ""},
        config=config,
    )
    response_dict = {}
    response_dict["Summary"] = response
    
    print(response_dict)

    # tokens_used += count_tokens_in_string(response)
    # et = time.time()
    # latency = et - st
    # cost = calculate_prompt_cost(prompt=review, model_name=os.getenv("BEDROCK_MODEL")) + \
    #        calculate_completion_cost(prompt = response, model_name=os.getenv("BEDROCK_MODEL"))
    # print(f"NUMBER OF TOKENS USED: {tokens_used}")
    # log_chat(session_id=secret_key, 
    #          user_feedback=review, 
    #          modification_needed = "",
    #          history="",
    #          model_response=response,
    #          timestamp=datetime.now(), 
    #          latency=latency, 
    #          tokens_used=tokens_used,
    #          cost = cost)
    return response, secret_key


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
    core_prompt = request.json["Core Prompt"]
    context_prompt = request.json["Context Prompt"]
    feedback_data = request.json['Feedback Data']
    context_prompt_0, context_prompt_1 = context_prompt.split("Context details:")[0], context_prompt.split("Context details:")[-1]
    final_json = dict()
    final_json["Overview"] = {}
    final_json["Responses"] = []
    # print(f"core_prompt:{core_prompt}")
    # print(f"context_prompt:{context_prompt}")
    # print(f"feedback_data:{feedback_data}")
    # print(f"context_prompt_0:{context_prompt_0}")
    # print(f"context_prompt_1:{context_prompt_1}")
    review_list = []
    for idx, data in enumerate(feedback_data):
        # print(data)
        try:       
            username = data.get("Customer")["Name"]
            # print(f"username : {username}")
        except:
            username = "None"
        user_id = data.get("ID")
        review = data.get("Review")["Text"]
        # print(f"Review: {review}")
        survey_data = data.get("Surveys")[0]["Questions"]
        # print(f"survey_data: {survey_data}")
        response_dict = {}
        feedback, original, secret_key = generate_response(username=username,
                                     user_id = user_id, 
                                     context_prompt_0= context_prompt_0,
                                     context_prompt_1=context_prompt_1,
                                     review=review, 
                                     core_prompt=core_prompt,
                                     survey_data = survey_data,
                                     secret_key=secret_key)
        # print(f"FEEDBACK IS LATEST: {feedback}")
        response_dict = feedback.copy()
        review_list.append(review)
        final_json["Responses"].append(response_dict)


    summary, secret_key = generate_consolidated_summary_response(
                                     review=review_list,
                                     secret_key=secret_key)
    # print(f"CONSOLIDATED SUMMARY: {summary}")


        # headers = {
        #     'X-Original': json.dumps({'original': original}),
        #     'X-Secret-Key': json.dumps({'secret_key': secret_key})
        # }
        # return Response(final_json, mimetype='text/event-stream', headers=headers)
    # summary["Sentiment"] = set([final_json["Responses"][e]["Sentiment"] for e in len(final_json["Responses"])])
    # Extract the unique sentiment
    unique_sentiment = list({response["Sentiment"] for response in final_json["Responses"]})

    # Check if there is only one unique sentiment
    if len(unique_sentiment) != 1:
        print("Warning: Multiple different sentiments found")

    # Extract and merge themes using list comprehension and set operations
    positive_themes = list({theme for response in final_json["Responses"] for theme in response["Positive_Themes"]})
    negative_themes = list({theme for response in final_json["Responses"] for theme in response["Negative_Themes"]})

    # Construct the final merged dictionary
    merged_data = {
        "Summary": summary,
        "Sentiment": unique_sentiment[0] if unique_sentiment else None,
        "Positive_Themes": positive_themes,
        "Negative_Themes": negative_themes
    }
    # print(f"MERGED DATA: {merged_data}")
    final_json["Overview"] = merged_data
    return jsonify(final_json)

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
    app.run(host='0.0.0.0', debug=False, port=5001)
