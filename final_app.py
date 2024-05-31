from langchain_aws import ChatBedrock
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import streamlit as st
import os
from dotenv import load_dotenv
import boto3
from botocore.config import Config


retry_config = Config(
    region_name='ap-south-1',
    retries={
        'max_attempts': 10,
        'mode': 'standard'
    }
)

# os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")
# ## Langmith tracking
# os.environ["LANGCHAIN_TRACING_V2"]="true"
# os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
session = boto3.Session()
boto3_bedrock_runtime = session.client(service_name='bedrock-runtime', 
                                       aws_access_key_id = os.environ.get("aws_access_key_id"),
                                       aws_secret_access_key = os.environ.get("aws_secret_access_key"),
                                       config=retry_config)
## Prompt Template



chat = ChatBedrock(
    model_id="meta.llama3-8b-instruct-v1:0",
    client=boto3_bedrock_runtime,
    model_kwargs={"temperature": 0.1},
)

messages = [
    HumanMessage(
        content="Translate this sentence from English to French. I love programming."
    )
]
chat.invoke(messages)

