# How to run
There should be an .env file in the same location as you app.py

The .env file will have following keys:

aws_access_key_id = 'YOUR AWS_ACCESS_KEY_ID' \
\
aws_secret_access_key = 'YOUR AWS_SECRET_ACCESS_KEY' \
\
region_name='YOUR REGION_NAME' \
\
LANGCHAIN_API_KEY = "YOU LANGCHAIN API KEY" \
\
LANGCHAIN_PROJECT = "YOU CAN GIVE THIS ANY NAME" \
\

# Local Testing
pip install -r requirements.txt \
\
python **app.py** \
\
python **test_api.py**

## Docker 

**sudo docker build -t quart-mistral-api .** \
\
**sudo docker run -p 5000:5000 quart-mistral-api**

