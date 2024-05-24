from quart import Quart, request, jsonify
import boto3
from langchain_community.llms import Bedrock
from langchain_aws import BedrockLLM
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from botocore.config import Config
import json

app = Quart(__name__)

# Configuring Boto3
retry_config = Config(
    region_name='ap-south-1',
    retries={
        'max_attempts': 10,
        'mode': 'standard'
    }
)

session = boto3.Session()
boto3_bedrock_runtime = session.client(service_name='bedrock-runtime', 
                                       aws_access_key_id = 'AKIAXRYOD7QBQVYTETXC',
                                       aws_secret_access_key = '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',
                                       config=retry_config) #creates a Bedrock client

model_kwargs = {
  "modelId": "mistral.mixtral-8x7b-instruct-v0:1",
  "contentType": "application/json",
  "accept": "application/json",
#   "body": "{\"prompt\":\"<s>[INST]I am going to Paris, what should I see?[/INST]\",\"max_tokens\":800,\"top_k\":50,\"top_p\":0.7,\"temperature\":0.7}"
}


llm = BedrockLLM(
    model_id="mistral.mixtral-8x7b-instruct-v0:1",
    client=boto3_bedrock_runtime,
    model_kwargs=model_kwargs,
)

# prompt = "What is the largest city in New Hampshire?" #the prompt to send to the model

# response_text = llm.invoke(prompt) #return a response to the prompt

# print(response_text)


# Instantiate the LLM with Bedrock


# Define the prompt template
template1 = '''I want you to act as a ecommerce customer support feeeback replier.
In a polite tone, respond to the product review given below:
REVIEW: {review}.
Make sure to format your response in a json format with the following keys.
    response: This will be the reponse text that you will provide based on the user review for the product'''

prompt1 = PromptTemplate(
    input_variables=['review'],
    template=template1
)

# input_prompt = prompt1.format_prompt(review="I am very happy with the product")

# # # Define the LLM chain
# # llm_chain = LLMChain(
# #     llm=llm,
# #     prompt=input_prompt
# # )


# resp = llm.invoke(input=input_prompt)
# review = "I am very happy with the product"
# response = llm_chain.arun(input_prompt)

# response

@app.route('/explain', methods=['POST'])
async def explain():
    data = await request.json
    review = data.get('review')
    print(review)
    input_prompt = prompt1.format_prompt(review=review)
    print("INPUT PROMPT IS:", input_prompt.text)
    if not review:
        return jsonify({'error': 'No review provided'}), 400

    try:
        response = llm.invoke(input=input_prompt.text)
        print(response)
        return jsonify({'response': response})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
