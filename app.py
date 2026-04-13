import streamlit as st
import torch
import numpy as np
import cv2
import tifffile as tiff
import matplotlib.pyplot as plt

from src.models.unet import UNet

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="Seismic Segmentation",
    layout="wide",
)

# ------------------ CUSTOM CSS ------------------
st.markdown("""
<style>
.main {
    background-color: #0e1117;
}
h1, h2, h3 {
    color: #ffffff;
}
.stButton>button {
    background-color: #4CAF50;
    color: white;
    border-radius: 8px;
    height: 3em;
    width: 100%;
}
</style>
""", unsafe_allow_html=True)

# ------------------ TITLE ------------------
st.title("Seismic Facies Segmentation")
st.caption("Deep learning model for geological interpretation")

# ------------------ LOAD MODEL ------------------
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(num_classes=10).to(device)
    model.load_state_dict(torch.load("models/best_model.pth", map_location=device))
    model.eval()
    return model, device

model, device = load_model()

# ------------------ PREPROCESS ------------------
def preprocess(file):
    img = tiff.imread(file)
    img = cv2.resize(img, (256, 256))
    img_norm = (img - np.mean(img)) / (np.std(img) + 1e-6)
    tensor = torch.tensor(img_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    return img, tensor

# ------------------ PREDICT ------------------
def predict(model, image, device):
    with torch.no_grad():
        output = model(image.to(device))
        pred = torch.argmax(output, dim=1).squeeze().cpu().numpy()
    return pred

# ------------------ SIDEBAR ------------------
with st.sidebar:
    st.header("Settings")
    st.write("Upload a seismic image to begin.")
    st.info("Supported: .tiff")

# ------------------ MAIN UI ------------------
uploaded = st.file_uploader("Upload Seismic Image", type=["tiff", "tif"])

if uploaded:
    st.success("File uploaded successfully")

    original, tensor = preprocess(uploaded)

    if st.button("Run Prediction"):
        with st.spinner("Running model..."):
            pred = predict(model, tensor, device)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Seismic Input")
            st.image(original, clamp=True)

        with col2:
            st.subheader("Predicted Facies")
            fig, ax = plt.subplots()
            ax.imshow(pred, cmap='jet')
            ax.axis('off')
            st.pyplot(fig)

        st.success("Prediction complete")