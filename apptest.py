import streamlit as st
from OmniGen import OmniGenPipeline
from PIL import Image
import os

def main():
    st.title("OmniGen Image Generator")

    # Sidebar for model loading
    st.sidebar.header("Model Configuration")
    model_name = st.sidebar.text_input("Enter model name:", "Shitao/OmniGen-v1")

    # Load the pipeline
    @st.cache_resource
    def load_pipeline(model_name, model_dir):
        return OmniGenPipeline.from_pretrained(model_name, model_path=model_dir)

    # Set the path to the directory containing all model files
    model_dir = "models/"  # Directory containing .safetensors and other files
    pipe = load_pipeline(model_name, model_dir=model_dir)

    # User input for prompt and parameters
    st.header("Image Generation")
    prompt = st.text_area("Enter your prompt:", 
                          "A highly realistic first-person perspective of someone recording themselves on a steep, rocky mountain slope.")

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
