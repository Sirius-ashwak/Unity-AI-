import streamlit as st
from PIL import Image
import io
import requests
import os

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

# Streamlit app title
st.title("Unity: Your Personal Assistant")

# Text input widget
text_input = st.text_input("Enter your query or description:")

# Create a container for the buttons
col1, col2 = st.columns([1, 1])

# Button for generating an image
with col1:
    if st.button("Generate Image"):
        if text_input:
            def query_image(payload):
                response = requests.post(IMAGE_API_URL, headers=IMAGE_HEADERS, json=payload)
                return response.content

            image_bytes = query_image({"inputs": text_input})
            image = Image.open(io.BytesIO(image_bytes))

            # Display the generated image
            st.image(image, caption='Generated Image', use_column_width=True)
        else:
            st.error("Please enter a description to generate an image.")

# Button for answering text
with col2:
    if st.button("Answer Me"):
        if text_input:
            import google.generativeai as genai
            genai.configure(api_key=GENAI_API_KEY)
            model = genai.GenerativeModel('gemini-pro')
            chat = model.start_chat(history=[])
            response = chat.send_message(text_input)

            # Display the response
            st.subheader("Response:")
            st.write(response.text)
        else:
            st.error("Please enter a query to get an answer.")
