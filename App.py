import streamlit as st
import google.generativeai as genai
from diffusers import DiffusionPipeline
from PIL import Image
import io

# Configure the API key for Google Generative AI
genai.configure(api_key="AIzaSyB3p-msB4w6fo5-tDmw-O-FkfOk9LiaYyw")

# Function to load the Diffusion Pipeline model
def load_pipeline():
    try:
        return DiffusionPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell")
    except ImportError:
        st.error("Transformers library is missing. Install it using 'pip install transformers'.")
    except Exception as e:
        st.error(f"Failed to load model: {e}")
    return None

# Load the Diffusion Pipeline model
pipeline = load_pipeline()

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
    
    # Image generation request
    if st.button("Generate Image"):
        if pipeline:
            try:
                # Generate an image
                image = pipeline(prompt=text).images[0]
                
                # Convert image to bytes
                buffer = io.BytesIO()
                image.save(buffer, format="PNG")
                buffer.seek(0)
                
                # Display the image
                st.image(buffer, caption="Generated Image", use_column_width=True)
            except Exception as e:
                st.error(f"Failed to generate image: {e}")
        else:
            st.error("Image generation pipeline is not available.")
