import warnings
warnings.filterwarnings(action = "ignore")
import boto3
import secrets
import json
import os
from botocore.config import Config
from pydantic import BaseModel, Field
from langchain_aws import ChatBedrock
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
import time

retry_config = Config(
                    region_name=os.environ.get("region_name"),
                    retries={
                        'max_attempts': 10,
                        'mode': 'standard'
                        }
                    )
    
session = boto3.Session()
boto3_bedrock_client = session.client(service_name='bedrock-runtime', 
                                       aws_access_key_id=os.environ.get("aws_access_key_id"),
                                       aws_secret_access_key=os.environ.get("aws_secret_access_key"),
                                       config=retry_config)
                                       
                                          
class ReviewsOutput(BaseModel):
        Response: str = Field(description="This will be the response that will be generate by you in reponse to the customer review.")
        Summary: str = Field(description="This will be the summary of the converstion based on the review provided.")
        Sentiment: str = Field(description="promoter or non-promoter")
        Positive_Themes: list = Field(description="This will be a python list of positives out of the entire conversation. Return a blank python list if there are no positives.")
        Negative_Themes: list = Field(description="This will be a python list of negatives out of the entire conversation. Return a blank python list if there are no negatives.")
    

def get_model():
    model = ChatBedrock(
        model_id = os.getenv("BEDROCK_MODEL"),
        client = boto3_bedrock_client,
        model_kwargs={
            "temperature": 0.5,
            "top_p": 0.9,
        },
    )
    return model 


def get_llm_response(model, 
                     username,
                     context_prompt_0,
                     context_prompt_1,
                     core_prompt,
                     survey_data,
                     review, 
                     output_parser):
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
    
    chain = prompt | model | output_parser
    
    return chain.invoke({"review": review, 
                         "survey_data":survey_data,
                         "input": f"""{context_prompt_1} 
                                      {context_prompt_0}
                                      Please adhere to the following policies:
                                      {core_prompt} 
                                      The username is {username} who provided the feedback."""
    })
    

def get_consolidated_summary(model, review):
    consolidated_summary_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                {review}.
                Summarize the response to capture the sentiments of the users in a maximum of two sentences.""",
            ),
            ("human", "{input}"),
        ]
    )
    
    output = StrOutputParser()
    
    consolidated_summary_runnable = consolidated_summary_prompt | model | output
    response = ""
    response =  consolidated_summary_runnable.invoke(
        {"review": f"""{review}""",
         "input": f"""Below are the reviews from {len(review)} user.
                      Summarize all the reviews into a single one to capture the sentiments of the all users in a maximum of 2 sentences.""",
         },
    )
    response_dict = {}
    response_dict["Summary"] = response
    return response


def lambda_handler(event, context):
    secret_key = secrets.token_hex(16)
    core_prompt = event.get("Core Prompt")
    context_prompt = event.get("Context Prompt")
    feedback_data = event.get('Feedback Data')
    bucket_name = os.environ.get('YOUR_BUCKET_NAME') 
    context_prompt_0, context_prompt_1 = context_prompt.split("Context details:")[0], context_prompt.split("Context details:")[-1]
    final_json = dict()
    final_json["Overview"] = {}
    final_json["Responses"] = []
    # print(f"core_prompt:{core_prompt}")
    # print(f"context_prompt:{context_prompt}")
    print(f"feedback_data:{feedback_data}")
    # print(f"context_prompt_0:{context_prompt_0}")
    # print(f"context_prompt_1:{context_prompt_1}")
    model = get_model()
    output_parser = JsonOutputParser(pydantic_object=ReviewsOutput)
    review_list = []
    for idx, data in enumerate(feedback_data):
        # print(data)
        try:       
            username = data.get("Customer")["Name"]
            # print(f"username : {username}")
        except:
            username = "None"
        feedback_id = data.get("ID")
        review = data.get("Review")["Text"]
        # print(f"Review: {review}")
        survey_data = data.get("Surveys")[0]["Questions"]
        print(f"survey_data: {survey_data}")
        response_dict = {}
        
        feedback = get_llm_response(model = model,
                                    username=username,
                                    context_prompt_0= context_prompt_0,
                                    context_prompt_1=context_prompt_1,
                                    review=review, 
                                    core_prompt=core_prompt,
                                    survey_data = survey_data,
                                    output_parser = output_parser)
        response_dict = feedback.copy()
        response_dict["feedback_id"] = feedback_id
        review_list.append(review)
        final_json["Responses"].append(response_dict)


    summary = get_consolidated_summary(model = model, review=review_list)
    print(f"CONSOLIDATED SUMMARY: {summary}")
    
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
    final_json["session_id"] = secret_key
    print(final_json)
    
    file_key = f'Data/{secret_key}.json'
    json_data = json.dumps(final_json)
    # file_content = 'This is the content of the file.' 
    s3 = boto3.client('s3') 
    status_code = s3.put_object(Bucket=bucket_name, Key=file_key, Body=json_data)
    print("STATUS CODE", status_code)
    return {
        'statusCode': 200,
        'body': final_json
    }

