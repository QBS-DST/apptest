import os
import requests
from OmniGen import OmniGenPipeline

def download_model(url, output_path):
    """Download the model file from SharePoint."""
    try:
        if not os.path.exists(output_path):
            print("Downloading model...")
            response = requests.get(url, stream=True)
            response.raise_for_status()  # Ensure the request was successful
            with open(output_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            print(f"Model downloaded to {output_path}")
        else:
            print(f"Model already exists at {output_path}")
    except requests.exceptions.RequestException as e:
        print(f"Error downloading the model: {e}")
        raise e

def main():
    import streamlit as st
    st.title("OmniGen Image Generator")

    # Model URL (replace with your direct download link)
    MODEL_URL = "https://qbslearning-my.sharepoint.com/download.aspx?p=harsh_saha/Emlm_8rxb-dAgXyyw81xCp8BpoVXpsISPemcU82tQY3dtw?e=v73T96"
    MODEL_PATH = "models/OmniGen-v1.safetensors"

    # Ensure the directory exists
    os.makedirs("models", exist_ok=True)

    # Download the model
    download_model(MODEL_URL, MODEL_PATH)

    # Load the pipeline
    @st.cache_resource
    def load_pipeline(model_path):
        return OmniGenPipeline.from_pretrained(model_path)

    pipe = load_pipeline(MODEL_PATH)

    # User input for prompt and parameters
    st.header("Image Generation")
    prompt = st.text_area("Enter your prompt:", "A beautiful landscape.")
    if st.button("Generate Image"):
        with st.spinner("Generating..."):
            image = pipe(prompt=prompt, height=512, width=512, guidance_scale=7.5)
            st.image(image[0], caption="Generated Image")

if __name__ == "__main__":
    main()
