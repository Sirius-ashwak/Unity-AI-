import requests
import os
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
API_URL = "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-large"
HEADERS = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}

def get_answer(query):
    """Fetches a response from the Hugging Face model based on the given query."""
    try:
        response = requests.post(API_URL, headers=HEADERS, json={"inputs": query})
        response.raise_for_status()  # Raise an error for failed requests
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching answer: {e}")
        return {"error": "Failed to fetch response"}
