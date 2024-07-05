import boto3
import json
import os
import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse, StreamingResponse
from pydantic import BaseModel
from typing import Optional
from botocore.config import Config

app = FastAPI()

app.mount("/demo", StaticFiles(directory="static", html=True))


@app.get("/")
async def root():
    return RedirectResponse(url='/demo/')


class reviewsInput(BaseModel):
    feedback: Optional[str] = None

@app.post("/api/write_with_ai")
def api_story(feedback_input: reviewsInput):
    if feedback_input.feedback == None or feedback_input.feedback == "":
        return None
    return StreamingResponse(bedrock_stream(feedback_input.feedback), media_type="text/html")

retry_config = Config(
    region_name=os.environ.get("region_name"),
    retries={
        'max_attempts': 10,
        'mode': 'standard'
    }
)

session = boto3.Session()
boto3_bedrock_runtime = session.client(service_name='bedrock-runtime', 
                                       aws_access_key_id=os.environ.get("aws_access_key_id"),
                                       aws_secret_access_key=os.environ.get("aws_secret_access_key"),
                                       config=retry_config)

async def bedrock_stream(feedback: str):
    instruction = f"""
    You are a world class Customer feedback response writer. Please write a feedback based on the feedback given by the customer.
    FEEDBACK: {feedback}.
    """

    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1024,
        "messages": [
            {
                "role": "user",
                "content": instruction,
            }
        ],
    })

    response = boto3_bedrock_runtime.invoke_model_with_response_stream(
            modelId=os.environ.get("BEDROCK_MODEL"),
            # accept="application/json",
            body=body,
            # contentType="application/json",
        )

    stream = response.get('body')
    if stream:
        for event in stream:
            chunk = event.get('chunk')
            if chunk:
                message = json.loads(chunk.get("bytes").decode())
                if message['type'] == "content_block_delta":
                    yield message['delta']['text'] or ""
                elif message['type'] == "message_stop":
                    yield "\n"

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8080")))