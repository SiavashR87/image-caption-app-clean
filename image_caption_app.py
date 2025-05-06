import os
import streamlit as st
import requests

# Configuration
API_URL = "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-base?pipeline_tag=image-to-text"
API_TOKEN = os.getenv("HF_API_TOKEN")
headers = {
    "Authorization": f"Bearer {API_TOKEN}",
    "Content-Type": "application/octet-stream"
}

def query(file_obj):
    # send raw bytes with correct pipeline tag
    data = file_obj.getvalue()
    response = requests.post(API_URL, headers=headers, data=data)
    try:
        return response.json(), response.status_code
    except ValueError:
        return {"error": response.text}, response.status_code

# Streamlit page setup
st.set_page_config(page_title="AI Image Caption Generator", page_icon="🖼️")
st.title("🖼️ AI Image Caption Generator")
st.write("Upload an image and I'll generate a caption using the BLIP model!")

# File uploader widget
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    # Check that secret is set
    if not API_TOKEN:
        st.error("🚨 HF_API_TOKEN not set. Go to your app's Settings → Secrets and add it there.")
    else:
        with st.spinner("Generating caption..."):
            result, status = query(uploaded_file)

        # Successful caption
        if status == 200 and isinstance(result, list) and "generated_text" in result[0]:
            st.success(f"Caption: {result[0]['generated_text']}")
        else:
            # HTTP or parsing error
            if isinstance(result, dict) and result.get("error"):  
                st.error(result["error"])
            else:
                st.error(f"Error (status {status}): {result}")

