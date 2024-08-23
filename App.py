import streamlit as st
import google.generativeai as genai
from PIL import Image
import requests
from io import BytesIO
import torch
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from diffusers import DiffusionPipeline  # Importing DiffusionPipeline

# Configure the API key
genai.configure(api_key="AIzaSyB3p-msB4w6fo5-tDmw-O-FkfOk9LiaYyw")

# Initialize the diffusion pipeline for image generation
pipeline = DiffusionPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell")

# Streamlit app title
st.title("Unity: Your Personal Assistant")

# Streamlit input widget with a friendly prompt
text = st.text_input("How can I assist you today?")

# Process the input if provided
if text:
    # Create the GenerativeModel object
    model = genai.GenerativeModel('gemini-pro')
    chat = model.start_chat(history=[])

    # Send the message and get the response
    response = chat.send_message(text)

    # Display the response with some styling
    st.subheader("Response:")
    st.write(response.text)

# Divider for image generation and detection
st.write("---")

# Image Generation using DiffusionPipeline
st.subheader("Generate an Image with DiffusionPipeline")
prompt = st.text_input("Enter a prompt for image generation with DiffusionPipeline:")

if prompt:
    st.write("Generating image with DiffusionPipeline...")
    try:
        # Generate the image using the pipeline
        image = pipeline(prompt).images[0]
        
        # Display the generated image
        st.image(image, caption="Generated Image with DiffusionPipeline", use_column_width=True)
    except Exception as e:
        st.write(f"An error occurred during image generation: {e}")

# Image Detection
st.subheader("Detect Objects in an Image")
uploaded_image = st.file_uploader("Upload an image for object detection:")

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Load a pretrained detection model
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()  # Set to evaluation mode

    # Transform the image for the model
    transform = transforms.Compose([transforms.ToTensor()])
    image_tensor = transform(image).unsqueeze(0)

    # Perform detection
    with torch.no_grad():
        predictions = model(image_tensor)

    # Process predictions
    for element in predictions[0]['boxes']:
        x1, y1, x2, y2 = element
        st.write(f"Detected object at [{x1:.2f}, {y1:.2f}, {x2:.2f}, {y2:.2f}]")

    st.write("Detection completed.")
