import requests

API_URL = "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-large"
headers = {"Authorization": "Bearer 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"}

def get_answer(query):
    response = requests.post(API_URL, headers=headers, json={"inputs": query})
    return response.json()
