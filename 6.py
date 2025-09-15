import streamlit as st
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import load_model
from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
from tf_keras_vis.utils.scores import CategoricalScore
import plotly.graph_objs as go

# Layout
st.set_page_config(layout="wide")
st.title("🩻 3D TB Detection Visualizer: Plain • TB Map • Overlay")

# Load model
@st.cache_resource
def load_tb_model():
    return load_model("tb_model.h5")

model = load_tb_model()
gradcam = Gradcam(model, model_modifier=ReplaceToLinear())
score = CategoricalScore([1])  # TB class

# Preprocess image
def preprocess(image):
    image = cv2.resize(image, (224, 224))
    image = image.astype(np.float32) / 255.
    return np.expand_dims(image, axis=0)

uploaded_file = st.file_uploader("📤 Upload Chest X-ray Image", type=['jpg', 'jpeg', 'png'])

if uploaded_file:
    # Load and process
    pil_img = Image.open(uploaded_file).convert("RGB")
    np_img = np.array(pil_img)
    input_tensor = preprocess(np_img)

    # Predict
    preds = model.predict(input_tensor)
    class_names = ['Normal', 'TB']
    predicted_class = class_names[np.argmax(preds)]
    st.success(f"🔍 Prediction: **{predicted_class}** ({np.max(preds)*100:.2f}%)")

    # Generate Grad-CAM
    cam = gradcam(score, input_tensor, penultimate_layer=-1)
    heatmap = np.uint8(255 * cam[0])
    heatmap_resized = cv2.resize(heatmap, (256, 256))

    # Prepare all images
    gray_img = cv2.resize(cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY), (256, 256)).astype(np.float32)
    overlay_img = gray_img + (heatmap_resized * 0.6)

    # Grid
    x = np.linspace(0, 1, 256)
    y = np.linspace(0, 1, 256)
    x, y = np.meshgrid(x, y)

    # 1. Plain X-ray 3D
    fig_plain = go.Figure(data=[
        go.Surface(z=gray_img, x=x, y=y, colorscale='Gray')
    ])
    fig_plain.update_layout(
        title="🩻 Plain X-ray (3D)",
        scene=dict(zaxis=dict(title='Pixel Intensity', range=[0, 255])),
        margin=dict(l=10, r=10, t=50, b=10),
        width=500
    )

    # 2. TB Heatmap Only
    fig_tb_only = go.Figure(data=[
        go.Surface(z=heatmap_resized.astype(np.float32), x=x, y=y, colorscale='Hot')
    ])
    fig_tb_only.update_layout(
        title="🔥 TB Activation Map (3D)",
        scene=dict(zaxis=dict(title='Heatmap Intensity', range=[0, 255])),
        margin=dict(l=10, r=10, t=50, b=10),
        width=500
    )

    # 3. Overlay 3D
    fig_overlay = go.Figure(data=[
        go.Surface(z=overlay_img, x=x, y=y, colorscale='Gray')
    ])
    fig_overlay.update_layout(
        title="🩻 + 🔥 Overlay (3D)",
        scene=dict(zaxis=dict(title='Combined Intensity')),
        margin=dict(l=10, r=10, t=50, b=10),
        width=500
    )

    # Display all 3
    col1, col2, col3 = st.columns(3)
    with col1:
        st.plotly_chart(fig_plain, use_container_width=True)
    with col2:
        st.plotly_chart(fig_tb_only, use_container_width=True)
    with col3:
        st.plotly_chart(fig_overlay, use_container_width=True)

else:
    st.info("📥 Upload a chest X-ray to visualize all 3D plots.")
