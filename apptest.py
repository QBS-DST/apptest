import os
import requests
import streamlit as st
from OmniGen import OmniGenPipeline
from PIL import Image

def download_file(url, output_path):
    """Download a file from OneDrive."""
    try:
        if not os.path.exists(output_path):
            st.write(f"Downloading {os.path.basename(output_path)}...")
            response = requests.get(url, stream=True)
            response.raise_for_status()
            with open(output_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            st.success(f"Downloaded to {output_path}")
        else:
            st.write(f"{os.path.basename(output_path)} already exists at {output_path}")
    except requests.exceptions.RequestException as e:
        st.error(f"Error downloading {os.path.basename(output_path)}: {e}")
        raise e

def main():
    st.title("OmniGen Image Generator")

    # Sidebar for model setup
    st.sidebar.header("Model Configuration")
    model_url = st.sidebar.text_input("Model File URL:", "https://qbslearning-my.sharepoint.com/:u:/p/harsh_saha/ETfBGQdsAv9DrnLuZqe-jawB6y2IBShEMibgYeyHdeSBOw?e=b8NBPr")
    diffusion_model_url = st.sidebar.text_input("Diffusion Model File URL:", "https://qbslearning-my.sharepoint.com/:u:/p/harsh_saha/EWQhiNGP8wRAoCDE5driXbYB-YAZO2XtJwmT4oP0ZqNGXA?e=VdKO97")
    config_url = st.sidebar.text_input("Config File URL:", "https://qbslearning-my.sharepoint.com/:u:/p/harsh_saha/Echeqm-E2FtKkXCu4FR-pv4Bkv2NaDf7cQNDGiofKoq-7g?e=8cYGvh")
    tokenizer_config_url = st.sidebar.text_input("Tokenizer Config URL:", "https://qbslearning-my.sharepoint.com/:u:/p/harsh_saha/EUqoLkPLthNFsrWY11Yp8XgBIIuHZjXyTaxUwIRkZpYo_g?e=E3cebp")
    special_tokens_map_url = st.sidebar.text_input("Special Tokens Map URL:", "https://qbslearning-my.sharepoint.com/:u:/p/harsh_saha/EalBk_vWjaJMiJ6-JW2BPdQBaet82VCceBG1PrY6IXqJZQ?e=ytGhqu")

    # Model directory and files
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    files_to_download = {
        "model.safetensors": model_url,
        "diffusion_pytorch_model.safetensors": diffusion_model_url,
        "config.json": config_url,
        "tokenizer_config.json": tokenizer_config_url,
        "special_tokens_map.json": special_tokens_map_url,
    }

    # Download required files
    for filename, url in files_to_download.items():
        if url.strip():
            download_file(url, os.path.join(model_dir, filename))

    # Load the pipeline
    @st.cache_resource
    def load_pipeline(model_dir):
        required_files = ["model.safetensors", "config.json", "tokenizer_config.json", "special_tokens_map.json"]
        for file in required_files:
            if not os.path.exists(os.path.join(model_dir, file)):
                raise FileNotFoundError(f"Required file {file} is missing in {model_dir}")

        return OmniGenPipeline.from_pretrained(model_dir)

    try:
        pipe = load_pipeline(model_dir)
        st.write("Pipeline loaded successfully.")
    except Exception as e:
        st.error(f"Failed to load pipeline: {e}")
        return

    # User input for image generation
    st.header("Image Generation")
    prompt = st.text_area(
        "Enter your prompt:",
        "A highly realistic first-person perspective of someone recording themselves on a steep, rocky mountain slope..."
    )
    height = st.slider("Image Height:", min_value=256, max_value=2048, value=1024, step=64)
    width = st.slider("Image Width:", min_value=256, max_value=2048, value=1920, step=64)
    guidance_scale = st.slider("Guidance Scale:", min_value=1.0, max_value=20.0, value=5.0, step=0.5)
    seed = st.number_input("Seed:", min_value=0, max_value=9999, value=111)

    # Generate and display image
    if st.button("Generate Image"):
        with st.spinner("Generating image..."):
            try:
                images = pipe(
                    prompt=prompt,
                    height=height,
                    width=width,
                    guidance_scale=guidance_scale,
                    seed=int(seed)
                )
                st.image(images[0], caption="Generated Image", use_column_width=True)

                # Save and download image
                save_path = os.path.join("outputs", "generated_image.png")
                os.makedirs("outputs", exist_ok=True)
                images[0].save(save_path)
                st.success(f"Image saved to {save_path}")
                with open(save_path, "rb") as file:
                    st.download_button(
                        label="Download Image",
                        data=file,
                        file_name="generated_image.png",
                        mime="image/png"
                    )
            except Exception as e:
                st.error(f"Error generating image: {e}")

if __name__ == "__main__":
    main()
