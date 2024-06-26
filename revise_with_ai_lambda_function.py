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
                     review, 
                     modifications,
                     output_parser):
    prompt = PromptTemplate(
    template="""{review}.
    {modifications}""",
    input_variables=["review", "modifications"],
    )
    print(prompt)
    chain = prompt | model | output_parser
    
    return chain.invoke({"review": review, "modifications": modifications})


def lambda_handler(event, context):
    # Extract the input parameters
    feedback_id = event['feedback_id']
    session_id = event['session_id']
    modifications = event['modifications']
    print(modifications)
    output_parser = StrOutputParser()
    model = get_model()
    # Define the S3 bucket and object key
    bucket_name = os.environ.get('YOUR_BUCKET_NAME') 
    file_key = f'Data/{session_id}.json'
    
    # Initialize the S3 client
    s3 = boto3.client('s3')
    
    try:
        # Download the JSON file from S3
        s3_response = s3.get_object(Bucket=bucket_name, Key=file_key)
        file_content = s3_response['Body'].read().decode('utf-8')
        json_data = json.loads(file_content)
        
        # Extract the feedback record
        feedback_record = None
        for response in json_data.get('Responses', []):
            if response.get('feedback_id') == feedback_id:
                feedback_record = response
                break
        
        if feedback_record is None:
            return {
                'statusCode': 404,
                'body': json.dumps('Feedback record not found')
            }
        
        print("FEEDBACK RECORD:", feedback_record)
        # Prepare the data to send to the LLM
        response_text = feedback_record.get('Response')
        
        print("ORIGINAL RESPONSE:", response_text)
        
        
        modified_response = get_llm_response(model = model,
                                             review = response_text,
                                             modifications = modifications,
                                             output_parser = output_parser)
        
        print("MODIFIED RESPONSE:", modified_response)
        
        feedback_record['Response'] = modified_response
        
        # Upload the updated JSON file back to S3
        updated_file_content = json.dumps(json_data)
        s3.put_object(Bucket=bucket_name, Key=file_key, Body=updated_file_content)
        return {
            'statusCode': 200,
            'body': modified_response
        }
        
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps(f'Error processing request: {str(e)}')
        }