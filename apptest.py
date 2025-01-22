import os
import requests
from OmniGen import OmniGenPipeline

def download_model(url, output_path):
    """Download the model file from OneDrive."""
    if not os.path.exists(output_path):
        print("Downloading model...")
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Ensure the request was successful
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:  # Skip keep-alive chunks
                    f.write(chunk)
        print(f"Model downloaded to {output_path}")
    else:
        print(f"Model already exists at {output_path}")

def main():
    # Streamlit app setup
    import streamlit as st
    st.title("OmniGen Image Generator")

    # Sidebar for configuration
    st.sidebar.header("Model Configuration")
    model_name = st.sidebar.text_input("Enter model name:", "Shitao/OmniGen-v1")

    # Model download URL from OneDrive (update with your direct link)
    MODEL_URL = "https://qbslearning-my.sharepoint.com/personal/p/harsh_saha/EnZxk8UoIGhJj3pr4ZaCr3sBffIJQvVfv3iDwlKRbLRCDQ?e=gYcDq7/download"
    MODEL_PATH = "models/OmniGen-v1.safetensors"

    # Ensure the directory exists
    os.makedirs("models", exist_ok=True)

    # Download the model if not already present
    download_model(MODEL_URL, MODEL_PATH)

    # Load the pipeline
    @st.cache_resource
    def load_pipeline(model_path):
        return OmniGenPipeline.from_pretrained(model_path)

    pipe = load_pipeline(MODEL_PATH)

    # User input for prompt
    st.header("Image Generation")
    prompt = st.text_area("Enter your prompt:", "A beautiful landscape.")
    if st.button("Generate Image"):
        with st.spinner("Generating..."):
            image = pipe(prompt=prompt, height=512, width=512, guidance_scale=7.5)
            st.image(image[0], caption="Generated Image")

if __name__ == "__main__":
    main()
