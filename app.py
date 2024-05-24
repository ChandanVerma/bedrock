from quart import Quart, request, jsonify
import boto3
from langchain_community.llms import Bedrock
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
  "body": "{\"prompt\":\"<s>[INST]I am going to Paris, what should I see?[/INST]\",\"max_tokens\":800,\"top_k\":50,\"top_p\":0.7,\"temperature\":0.7}"
}


llm = Bedrock(
    model_id="mistral.mixtral-8x7b-instruct-v0:1",
    client=boto3_bedrock_runtime,
    # model_kwargs=model_kwargs,
)

# prompt = "What is the largest city in New Hampshire?" #the prompt to send to the model

# response_text = llm.invoke(prompt) #return a response to the prompt

# print(response_text)


# Instantiate the LLM with Bedrock


# Define the prompt template
template1 = '''I want you to act as a acting dietician for people.
In an easy way, explain the benefits of {review}.'''

prompt1 = PromptTemplate(
    input_variables=['review'],
    template=template1
)

# # Define the LLM chain
# llm_chain = LLMChain(
#     llm=llm,
#     prompt=prompt1
# )


llm.invoke()
review = "I am very happy with the product"
response = llm_chain.arun(review)

response

@app.route('/explain', methods=['POST'])
async def explain():
    data = await request.json
    review = data.get('review')
    if not review:
        return jsonify({'error': 'No review provided'}), 400

    try:
        response = await llm_chain.arun(review)
        return jsonify({'response': response})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
