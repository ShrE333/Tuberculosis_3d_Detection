import streamlit as st
import cv2
import numpy as np
import plotly.graph_objs as go
from PIL import Image
from tensorflow.keras.models import load_model
from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
from tf_keras_vis.utils.scores import CategoricalScore

# Page configuration
st.set_page_config(layout="wide")
st.title(" TB Chest X-ray Visualizer")
st.markdown("---")

# Load model and setup GradCAM
model = load_model("tb_model.h5")
gradcam = Gradcam(model, model_modifier=ReplaceToLinear())

# Utility: preprocess input image
def preprocess_image(image):
    image = cv2.resize(image, (224, 224))
    image = image.astype(np.float32) / 255.0
    return np.expand_dims(image, axis=0)

# Utility: create a 3D surface plot
def create_3d_plot(z_data, title):
    z = z_data.astype(np.float32)
    x = np.linspace(0, 1, z.shape[1])
    y = np.linspace(0, 1, z.shape[0])
    x, y = np.meshgrid(x, y)
    fig = go.Figure(data=[go.Surface(z=z, x=x, y=y, colorscale='Gray')])
    fig.update_layout(
        title=title,
        scene=dict(zaxis=dict(title='Intensity', range=[0, 255])),
        margin=dict(r=20, l=20, b=20, t=40),
        width=400,
        height=400
    )
    return fig

# Upload image
uploaded_file = st.file_uploader(" Upload Chest X-ray Image", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    # Convert and preprocess image
    image_pil = Image.open(uploaded_file).convert("RGB")
    img = np.array(image_pil)
    img_resized = cv2.resize(img, (224, 224))

    # Prediction
    pred = model.predict(preprocess_image(img_resized))
    class_names = ['Normal', 'TB Detected']
    predicted_class = class_names[np.argmax(pred)]
    confidence = round(100 * np.max(pred), 2)

    # Display prediction results
    st.markdown("###  Model Prediction")
    st.success(f"**Result:** `{predicted_class}`  \n**Confidence:** `{confidence}%`")
    st.markdown("---")

    # GradCAM Heatmap generation
    cam = gradcam(CategoricalScore([1]), preprocess_image(img_resized), penultimate_layer=-1)
    heatmap = cam[0]
    heatmap_norm = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap))  # normalize
    heatmap_2d = np.uint8(255 * heatmap_norm)

    # Overlay heatmap
    overlay = cv2.applyColorMap(heatmap_2d, cv2.COLORMAP_JET)
    overlayed_img = cv2.addWeighted(img_resized, 0.6, overlay, 0.4, 0)

    # Columns layout for results
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📸 Original X-ray (Grayscale)")
        gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
        st.image(gray, use_column_width=True, clamp=True)
        st.plotly_chart(create_3d_plot(gray, "3D X-ray"), use_container_width=True)

    with col2:
        st.subheader("🔥 Heatmap Overlay")
        st.image(overlayed_img, use_column_width=True, clamp=True)
        overlay_gray = cv2.cvtColor(overlayed_img, cv2.COLOR_BGR2GRAY)
        st.plotly_chart(create_3d_plot(overlay_gray, "3D Heatmap Overlay"), use_container_width=True)
else:
    st.info("👆 Upload a chest X-ray to begin the analysis.")
