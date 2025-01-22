import streamlit as st
from OmniGen import OmniGenPipeline
from PIL import Image

def main():
    # Application Title
    st.title("OmniGen Image Generator")

    # Sidebar for Model Configuration
    st.sidebar.header("Model Configuration")
    model_name = st.sidebar.text_input("Enter model name:", "Shitao/OmniGen-v1")

    # Load the pipeline
    @st.cache_resource
    def load_pipeline(model_name):
        """
        Load the OmniGen pipeline from Hugging Face model hub.
        Uses Streamlit's caching to avoid reloading unnecessarily.
        """
        return OmniGenPipeline.from_pretrained(model_name)

    # Initialize the pipeline
    pipe = load_pipeline(model_name)

    # User Input for Image Generation
    st.header("Image Generation")
    prompt = st.text_area(
        "Enter your prompt:",
        "A highly realistic first-person perspective of someone recording themselves on a steep, rocky mountain slope. "
        "The scene captures the rugged texture of the sedimentary rocks, scattered with small debris and fossils, "
        "with one leg visibly slipping down the slope. The background includes a dramatic valley view, with sunlight "
        "illuminating sections of the terrain while other areas remain in shadow. Sparse greenery is present along the "
        "edges of the rocks, and the perspective conveys the tension and precariousness of the moment, emphasizing the "
        "danger of navigating this rugged mountain environment."
    )

    # Image parameters
    height = st.slider("Image Height:", min_value=256, max_value=2048, value=1024, step=64)
    width = st.slider("Image Width:", min_value=256, max_value=2048, value=1920, step=64)
    guidance_scale = st.slider("Guidance Scale:", min_value=1.0, max_value=20.0, value=5.0, step=0.5)
    seed = st.number_input("Seed:", min_value=0, max_value=9999, value=111)

    # Image Generation Button
    if st.button("Generate Image"):
        generate_and_display_image(pipe, prompt, height, width, guidance_scale, seed)


def generate_and_display_image(pipe, prompt, height, width, guidance_scale, seed):
    """
    Generate an image using the provided pipeline and parameters,
    then display and save the result.
    """
    with st.spinner("Generating image..."):
        # Generate image using the pipeline
        images = pipe(
            prompt=prompt,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            seed=int(seed)
        )

        # Display the generated image
        st.image(images[0], caption="Generated Image", use_column_width=True)

        # Save the generated image locally
        save_path = "generated_image.png"
        images[0].save(save_path)
        st.success(f"Image saved as {save_path}")

        # Add a download button for the image
        with open(save_path, "rb") as file:
            st.download_button(
                label="Download Image",
                data=file,
                file_name="generated_image.png",
                mime="image/png"
            )


if __name__ == "__main__":
    main()
