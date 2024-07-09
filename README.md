# How to run
There should be an .env file in the same location as your prod_flask.py

The .env file will have following keys:

aws_access_key_id = 'YOUR AWS_ACCESS_KEY_ID' \
\
aws_secret_access_key = 'YOUR AWS_SECRET_ACCESS_KEY' \
\
region_name='YOUR REGION_NAME' \
\
BEDROCK_MODEL = "mistral.mistral-7b-instruct-v0:2" \
\
YOUR_BUCKET_NAME = "YOUR S3 BUCKET NAME" \


# Local Testing
pip install -r requirements.txt \
\
python **prod_flask.py**


## Docker Compose

**docker-compose up --build**

## API HOSTED ON :

http://127.0.0.1:5001/write_with_ai

## FOR LAMBDA FUNCTIONS
1. Create a Lambda function in AWS
2. Copy the Lambda function code from Lmabda_function.py and paste into the code section of your AWS lambda function
3. Add the environment variables as mentioned above in the configuration -> environment variables tab of your AWS Lambda function page
4. Increase the timeout to 1 minute in AWS lambda in the configuration -> General Configuration tab
5. Scroll down in the AWS lambda code window and move to the last section to add layers. Click on add layers -> specify ARN and paste arn:aws:lambda:us-east-1:770693421928:layer:Klayers-p311-boto3:11
6. Add another layer and this time click on create new layer in the LAYER SOURCE section within choose a layer and upload the given zip file.

### FOR STREAMING
1. Install SAM CLI https://docs.aws.amazon.com/serverless-application-model/latest/developerguide/install-sam-cli.html
2. make changes in the template.yaml (secret key, region name, model name, access key)
3. add .env inside the app folder
4. execute the command:  "sam build --use-container"
5. execute the command: "sam deploy --guided"
6. you will get a URL at the END something like: https://cxle0apxvl.lambda-url.ap-south-1.on.aws
