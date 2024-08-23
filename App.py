import streamlit as st
from PIL import Image
import io
import requests

# Set up API URLs and keys
GENAI_API_KEY = "AIzaSyB3p-msB4w6fo5-tDmw-O-FkfOk9LiaYyw"
IMAGE_API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
IMAGE_HEADERS = {"Authorization": f"Bearer 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"}

# Streamlit app title
st.title("Unity: Your Personal Assistant")

# Text input widget
text_input = st.text_input("Enter your query or description:")

# Create a container for the buttons
col1, col2 = st.columns([1, 1])

# Button for generating image
with col1:
    if st.button("Generate Image"):
        if text_input:
            # Generate image using the provided description
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
