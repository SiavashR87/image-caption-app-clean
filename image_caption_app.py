import os
import streamlit as st
import requests

# Configuration
# Use the raw model endpoint (no pipeline); the model auto-detects image-to-text
API_URL = "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-base"
API_TOKEN = os.getenv("HF_API_TOKEN")
headers = {"Authorization": f"Bearer {API_TOKEN}"}

# Function to call the Hugging Face inference API with file upload
def query(file_obj):
    files = {"file": file_obj.getvalue()}
    response = requests.post(API_URL, headers=headers, files=files)
    status = response.status_code
    try:
        data = response.json()
    except ValueError:
        data = response.text
    return data, status

# Streamlit page setup
st.set_page_config(page_title="AI Image Caption Generator", page_icon="🖼️")
st.title("🖼️ AI Image Caption Generator")
st.write("Upload an image and I'll generate a caption using the BLIP model!")

# File uploader widget
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    if not API_TOKEN:
        st.error("🚨 HF_API_TOKEN not set. Go to your app's Settings → Secrets and add it there.")
    else:
        with st.spinner("Generating caption..."):
            result, status = query(uploaded_file)

        if status == 200 and isinstance(result, list) and "generated_text" in result[0]:
            st.success(f"Caption: {result[0]['generated_text']}")
        else:
            # Show status and response text for diagnosis
            st.error(f"Error {status}: {result}")
if uploaded_file:
    st.write("### Debug Info")
    st.write("If you encounter issues, please check the following:")
    st.write("- Ensure your image is in a supported format (PNG, JPG, JPEG).")
    st.write("- Check your API token and model endpoint.")
    st.write("- If the error persists, please contact support with the details above.")
    # For debugging locally, you can print logs to console:
    st.write("\n--- Debug Info ---")
    st.write(f"Endpoint: {API_URL}")
    st.write(f"Status: {status if 'status' in locals() else 'n/a'}")
    st.write(f"Response: {result if 'result' in locals() else 'n/a'}")