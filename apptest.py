import os
import requests
import streamlit as st
from OmniGen import OmniGenPipeline
from PIL import Image

def download_model(url, output_path):
    """Download the model file from OneDrive."""
    try:
        if not os.path.exists(output_path):
            st.write("Downloading model from OneDrive...")
            response = requests.get(url, stream=True)
            response.raise_for_status()  # Ensure the request was successful
            with open(output_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            st.success(f"Model downloaded to {output_path}")
        else:
            st.write(f"Model already exists at {output_path}")
    except requests.exceptions.RequestException as e:
        st.error(f"Error downloading the model: {e}")
        raise e

def main():
    st.title("OmniGen Image Generator")

    # Sidebar for model loading
    st.sidebar.header("Model Configuration")
    model_name = st.sidebar.text_input("Enter model name:", "Shitao/OmniGen-v1")

    # OneDrive direct download link for the model
    MODEL_URL = "https://qbslearning-my.sharepoint.com/:u:/p/harsh_saha/ETfBGQdsAv9DrnLuZqe-jawB6y2IBShEMibgYeyHdeSBOw?e=d11YHE"
    MODEL_PATH = "models/model.safetensors"

    # Ensure the directory exists
    os.makedirs("models", exist_ok=True)

    # Download the model if not already downloaded
    download_model(MODEL_URL, MODEL_PATH)

    # Load the pipeline
    @st.cache_resource
    def load_pipeline(model_path):
        return OmniGenPipeline.from_pretrained(model_path)

    pipe = load_pipeline(MODEL_PATH)

    # User input for prompt and parameters
    st.header("Image Generation")
    prompt = st.text_area("Enter your prompt:", 
                          "A highly realistic first-person perspective of someone recording themselves on a steep, rocky mountain slope. The scene captures the rugged texture of the sedimentary rocks, scattered with small debris and fossils, with one leg visibly slipping down the slope. The background includes a dramatic valley view, with sunlight illuminating sections of the terrain while other areas remain in shadow. Sparse greenery is present along the edges of the rocks, and the perspective conveys the tension and precariousness of the moment, emphasizing the danger of navigating this rugged mountain environment.")

    height = st.slider("Image Height:", min_value=256, max_value=2048, value=1024, step=64)
    width = st.slider("Image Width:", min_value=256, max_value=2048, value=1920, step=64)
    guidance_scale = st.slider("Guidance Scale:", min_value=1.0, max_value=20.0, value=5.0, step=0.5)
    seed = st.number_input("Seed:", min_value=0, max_value=9999, value=111)

    # Generate and display image
    if st.button("Generate Image"):
        with st.spinner("Generating image..."):
            images = pipe(
                prompt=prompt,
                height=height,
                width=width,
                guidance_scale=guidance_scale,
                seed=int(seed)
            )

            # Display the generated image
            st.image(images[0], caption="Generated Image", use_column_width=True)

            # Save option
            save_path = "generated_image.png"
            images[0].save(save_path)
            st.success(f"Image saved as {save_path}")
            with open(save_path, "rb") as file:
                btn = st.download_button(
                    label="Download Image",
                    data=file,
                    file_name="generated_image.png",
                    mime="image/png"
                )

if __name__ == "__main__":
    main()
