import requests

def test_api():
    url = "http://localhost:5000/explain"
    payload = {"review": "I am very happy with the product"}
    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(url, json=payload, headers=headers)
        print(response)
        response.raise_for_status()
        print("Response Status Code:", response.status_code)
        print("Response JSON:", response.json())
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")


def test_stream(survey_id= 7885020, user_review=None):
    response = requests.post(
        'http://localhost:5000/write_with_ai',
        json={
            'survey_id': survey_id,
            'comment': user_review
        },
        stream=True
    )
    import time
    for line in response.iter_lines():
        if line:
            print(line.decode('utf-8'))
            time.sleep(0.5)

if __name__ == "__main__":
    test_api()
    test_stream(survey_id=7885020, user_review="I am very happy with the product")