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
from langchain_core.runnables import RunnableSequence
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
        Default: str = Field(description="This will be the response that will be generate by you in reponse to the customer review.")
        More_friendly: str = Field(description="This will be the response that will be generate by you in reponse to the customer review in a friendly way.")
        More_Professional: str = Field(description="This will be the response that will be generate by you in reponse to the customer review in a more professional way.")
        More_Concise: list = Field(description="This will be the response that will be generate by you in reponse to the customer review in a more concise way.")
        

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
                Based on the Survey data provided generate 4 different summary of the conversation,
                1. The first summary should be the default one that you would generally repond with.
                2. The second summary should be more friendly
                3. The third summary should be more professional
                4. The fourth summary should be more concise
                {survey_data}
                format you response in a json format with the following keys:
                Default: This will be the response that will be generate by you in reponse to the customer review.
                More friendly: This will be the response that will be generate by you in reponse to the customer review in a friendly way.
                More Professional: This will be the response that will be generate by you in reponse to the customer review in a more professional way.
                More Concise: This will be the response that will be generate by you in reponse to the customer review in a more concise way.
                
                Recheck the response is strictly in a json format with the following keys and nothings else:
                Default, More friendly, More Professional, More Concise""",
    input_variables=["review", "survey_data"],
    partial_variables={"format_instructions": output_parser.get_format_instructions()},
    )
    
    chain = prompt | model | output_parser
    
    batch_input = [
        {
        "review": review,
        "survey_data": survey,
        "input": f"""{context_prompt_1} 
                     {context_prompt_0}
                     Please adhere to the following policies:
                     {core_prompt} 
                     The username is {username} who provided the feedback."""
        }
    for review, survey, username in zip(review, survey_data, username)
    ]
    
    resp = chain.batch(batch_input)
    return resp
    

# def get_consolidated_summary(model, review):
#     consolidated_summary_prompt = ChatPromptTemplate.from_messages(
#         [
#             (
#                 "system",
#                 """
#                 {review}.
#                 Summarize the response to capture the sentiments of the users in a maximum of two sentences.""",
#             ),
#             ("human", "{input}"),
#         ]
#     )
    
#     output = StrOutputParser()
    
#     consolidated_summary_runnable = consolidated_summary_prompt | model | output
#     response = ""
#     response =  consolidated_summary_runnable.invoke(
#         {"review": f"""{review}""",
#          "input": f"""Below are the reviews from {len(review)} user.
#                       Summarize all the reviews into a single one to capture the sentiments of the all users in a maximum of 2 sentences.""",
#          },
#     )
#     response_dict = {}
#     response_dict["Summary"] = response
#     return response


def lambda_handler(event, context):
    secret_key = secrets.token_hex(16)
    core_prompt = event.get("Core Prompt")
    context_prompt = event.get("Context Prompt")
    feedback_data = event.get('Feedback Data')
    bucket_name = os.environ.get('YOUR_BUCKET_NAME') 
    context_prompt_0, context_prompt_1 = context_prompt.split("Context details:")[0], context_prompt.split("Context details:")[-1]
    final_json = dict()
    final_json["Overview"] = {}

    reviews = [feedback["Review"]["Text"] for feedback in feedback_data]
    survey_questions = [survey["Questions"] for feedback in feedback_data for survey in feedback["Surveys"]]
    user_name = [feedback["Customer"]["Name"] if feedback["Customer"].get("Name") is not None else None for feedback in feedback_data]
    feedback_ids = [feedback["ID"] for feedback in feedback_data]
    print("FEEDBACK ID:", feedback_ids)
    print("USER NAME:", user_name)
    
    model = get_model()
    output_parser = JsonOutputParser(pydantic_object=ReviewsOutput)
    
    feedback = get_llm_response(model = model,
                                username=user_name,
                                context_prompt_0 = context_prompt_0,
                                context_prompt_1 = context_prompt_1,
                                review=reviews, 
                                core_prompt = core_prompt,
                                survey_data = survey_questions,
                                output_parser = output_parser)
    
    final_json["Responses"] = feedback
    # Add feedback_id to each response
    for response, feedback_id in zip(final_json["Responses"], feedback_ids):
        response["feedback_id"] = feedback_id
        
    # review_list = [item["Response"] for item in feedback]

    # summary = get_consolidated_summary(model = model, review=review_list)
    # unique_sentiment = list({response["Sentiment"] for response in final_json["Responses"]})

    # Check if there is only one unique sentiment
    # if len(unique_sentiment) != 1:
    #     print("Warning: Multiple different sentiments found")

    # # Extract and merge themes using list comprehension and set operations
    # positive_themes = list({theme for response in final_json["Responses"] for theme in response["Positive_Themes"]})
    # negative_themes = list({theme for response in final_json["Responses"] for theme in response["Negative_Themes"]})

    # Construct the final merged dictionary
    # merged_data = {
    #     "Summary": summary,
    #     "Sentiment": unique_sentiment[0] if unique_sentiment else None,
    #     "Positive_Themes": positive_themes,
    #     "Negative_Themes": negative_themes
    # }
    # final_json["Overview"] = merged_data
    # final_json["session_id"] = secret_key
    
    
    ##### UNCOMMENT THIS SECTION IF YOU WANT TO UPLOAD RESULTS TO S3 #####
    
    # file_key = f'Data/{secret_key}.json'
    # json_data = json.dumps(final_json)
    # s3 = boto3.client('s3') 
    # status_code = s3.put_object(Bucket=bucket_name, Key=file_key, Body=json_data)

    return {
        'statusCode': 200,
        'body': final_json
    }

