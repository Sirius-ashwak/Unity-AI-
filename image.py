import requests
import io
from PIL import Image

API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
headers = {"Authorization": "Bearer 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"}

def generate_image(description):
    def query(payload):
        response = requests.post(API_URL, headers=headers, json=payload)
        return response.content

    image_bytes = query({"inputs": description})
    image = Image.open(io.BytesIO(image_bytes))
    
    return image
