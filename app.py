import warnings
warnings.filterwarnings(action="ignore")
from quart import Quart, request, jsonify
import boto3
import os
from langchain_community.llms import Bedrock
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from botocore.config import Config
from dotenv import load_dotenv
load_dotenv(".env")

app = Quart(__name__)

# Configuring Boto3
retry_config = Config(
    region_name=os.environ.get("region_name"),
    retries={
        'max_attempts': 10,
        'mode': 'standard'
    }
)

session = boto3.Session()
boto3_bedrock_runtime = session.client(service_name='bedrock-runtime', 
                                       aws_access_key_id = os.environ.get("aws_access_key_id"),
                                       aws_secret_access_key = os.environ.get("aws_secret_access_key"),
                                       config=retry_config) #creates a Bedrock client

def get_text_response(input_content): #text-to-text client function

    llm = Bedrock(
    model_id="mistral.mistral-7b-instruct-v0:2",
    client=boto3_bedrock_runtime,
    # model_kwargs=model_kwargs,
    )

    # Define the prompt template
    template1 = '''[INST] I want you to act as a customer support feeeback replier.
    In a polite tone, respond to the product review given below:
    REVIEW: {review}.
    DONOT PROVIDE ANY KIND OF PLACEHOLDERS like [Your Name] or anything like this IN YOUR RESPONSE
    [/INST]'''

    prompt = PromptTemplate(
        input_variables=['review'],
        template=template1
    )

    llm = LLMChain(llm=llm, prompt=prompt)

    return llm.invoke({"review": input_content})["text"].strip()

# resp = get_text_response("i am very happy with the product")

@app.route('/explain', methods=['POST'])
async def explain():
    data = await request.json
    review = data.get('review')
    
    if not review:
        return jsonify({'error': 'No review provided'}), 400

    try:
        response = get_text_response(review)
        # print(response)
        return jsonify({'response': response})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)