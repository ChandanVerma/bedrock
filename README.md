# How to run
There should be an .env file in the same location as you prod_flask.py

The .env file will have following keys:

aws_access_key_id = 'YOUR AWS_ACCESS_KEY_ID' \
\
aws_secret_access_key = 'YOUR AWS_SECRET_ACCESS_KEY' \
\
region_name='YOUR REGION_NAME' \
\
BEDROCK_MODEL = "mistral.mistral-7b-instruct-v0:2"


# Local Testing
pip install -r requirements.txt \
\
python **prod_flask.py**


## Docker Compose

**docker-compose up --build**

## API HOSTED ON :

http://127.0.0.1:5001/write_with_ai
