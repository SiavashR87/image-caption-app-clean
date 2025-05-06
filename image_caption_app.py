import streamlit as st
import requests

API_URL = "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-base"
API_TOKEN = os.getenv("HF_API_TOKEN")
headers = {"Authorization": f"Bearer {API_TOKEN}"}

def query(file):
    response = requests.post(API_URL, headers=headers, data=file.getvalue())
    try:
        return response.json()
    except Exception as e:
        return {"error": f"Failed to parse response: {e}"}

st.set_page_config(page_title="AI Image Caption Generator", page_icon="🖼️")
st.title("🖼️ AI Image Caption Generator")
st.write("Upload an image, and I'll generate a caption for you using the BLIP model!")

uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    st.write("Generating caption...")

    result = query(uploaded_file)

    if isinstance(result, list) and "generated_text" in result[0]:
        st.success(f"Caption: {result[0]['generated_text']}")
    elif "error" in result:
        st.error(result["error"])
    else:
        st.error(f"Unexpected response: {result}")
        