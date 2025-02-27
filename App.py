import streamlit as st
from PIL import Image
import io
import requests
import os
import google.generativeai as genai

# Load environment variables locally (ignored in Render)
if os.getenv("RENDER") is None:
    from dotenv import load_dotenv
    load_dotenv()

# Fetch API keys from environment variables
GENAI_API_KEY = os.getenv("GOOGLE_GENERATIVEAI_API_KEY")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

# Check if API keys are available
if not GENAI_API_KEY or not HUGGINGFACE_API_KEY:
    st.error("Missing API keys! Please set them in your environment variables.")
    st.stop()

# Set up API URLs and headers
IMAGE_API_URL = "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-dev"
IMAGE_HEADERS = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}

# Configure Gemini API
genai.configure(api_key=GENAI_API_KEY)
MODEL_NAME = "models/gemini-2.0-pro-exp-02-05"

def generate_image(description):
    """Fetch an AI-generated image based on the input description."""
    response = requests.post(IMAGE_API_URL, headers=IMAGE_HEADERS, json={"inputs": description})
    if response.status_code == 200:
        return Image.open(io.BytesIO(response.content))
    else:
        st.error("Error generating image. Please try again.")
        return None

def generate_text_response(query):
    """Fetch a text-based response from Gemini AI."""
    try:
        model = genai.GenerativeModel(MODEL_NAME)
        response = model.generate_content(query)
        return response.text
    except Exception as e:
        st.error(f"Error generating response: {e}")
        return None

# Streamlit app UI
st.title("Unity: Your Personal Assistant")

text_input = st.text_input("Enter your query or description:")
col1, col2 = st.columns([1, 1])

# Button for generating an image
with col1:
    if st.button("Generate Image"):
        if text_input:
            image = generate_image(text_input)
            if image:
                st.image(image, caption='Generated Image', use_column_width=True)
        else:
            st.error("Please enter a description to generate an image.")

# Button for answering text
with col2:
    if st.button("Answer Me"):
        if text_input:
            response = generate_text_response(text_input)
            if response:
                st.subheader("Response:")
                st.write(response)
        else:
            st.error("Please enter a query to get an answer.")
