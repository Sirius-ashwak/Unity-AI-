import requests
import io
import os
from PIL import Image
from dotenv import load_dotenv

# Load environment variables locally (ignored in Render)
if os.getenv("RENDER") is None:
    load_dotenv()

# Fetch Hugging Face API key from environment variables
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

# Check if API key is available
if not HUGGINGFACE_API_KEY:
    raise ValueError("Missing Hugging Face API key! Please set it in your environment variables.")

# Set up API URL and headers
API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
HEADERS = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}

def generate_image(description):
    """Generates an image based on the given description using Hugging Face API."""
    def query(payload):
        response = requests.post(API_URL, headers=HEADERS, json=payload)
        response.raise_for_status()  # Raise error for failed requests
        return response.content

    try:
        image_bytes = query({"inputs": description})
        image = Image.open(io.BytesIO(image_bytes))
        return image
    except requests.exceptions.RequestException as e:
        print(f"Error generating image: {e}")
        return None
