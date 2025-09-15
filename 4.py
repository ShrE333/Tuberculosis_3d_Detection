import streamlit as st
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import load_model
from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
from tf_keras_vis.utils.scores import CategoricalScore
import plotly.graph_objs as go

# Setup
st.set_page_config(layout="wide")
st.title("🧠 3D TB Visualizer with Heatmap Overlay")

# Load model
@st.cache_resource
def load_tb_model():
    return load_model("tb_model.h5")

model = load_tb_model()
gradcam = Gradcam(model, model_modifier=ReplaceToLinear())
score = CategoricalScore([1])  # TB class

# Preprocessing
def preprocess(image):
    image = cv2.resize(image, (224, 224))
    image = image.astype(np.float32) / 255.
    return np.expand_dims(image, axis=0)

uploaded_file = st.file_uploader("📤 Upload Chest X-ray Image", type=['jpg', 'jpeg', 'png'])

if uploaded_file:
    # Load image
    pil_img = Image.open(uploaded_file).convert("RGB")
    np_img = np.array(pil_img)
    st.image(np_img, caption="🩻 Uploaded X-ray", use_column_width=True)

    # Preprocess
    input_tensor = preprocess(np_img)

    # Predict
    preds = model.predict(input_tensor)
    class_names = ['Normal', 'TB']
    predicted_class = class_names[np.argmax(preds)]
    st.success(f"🔍 Prediction: **{predicted_class}** ({np.max(preds)*100:.2f}%)")

    # Grad-CAM
    cam = gradcam(score, input_tensor, penultimate_layer=-1)
    heatmap = np.uint8(255 * cam[0])
    heatmap_resized = cv2.resize(heatmap, (256, 256))

    # X-ray base (grayscale)
    gray_img = cv2.resize(cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY), (256, 256)).astype(np.float32)

    # Merge heatmap into grayscale
    overlay_img = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
    overlay_img = cv2.cvtColor(overlay_img, cv2.COLOR_BGR2RGB)
    overlay_img = cv2.addWeighted(np.stack([gray_img]*3, axis=-1).astype(np.uint8), 0.7, overlay_img, 0.3, 0)

    # Create merged surface Z using the intensity + weighted heatmap
    merged_surface = gray_img + (heatmap_resized * 0.6)

    # Create 3D axes
    x = np.linspace(0, 1, 256)
    y = np.linspace(0, 1, 256)
    x, y = np.meshgrid(x, y)

    # Show 2D merged preview
    st.subheader("🖼️ Heatmap Overlay on 2D X-ray")
    st.image(overlay_img, use_column_width=True)

    # 3D Visualization with heatmap
    st.subheader("🌐 3D View of TB Region on Full X-ray")
    fig_overlay = go.Figure(data=[go.Surface(z=merged_surface, x=x, y=y, colorscale='Hot')])
    fig_overlay.update_layout(
        title="🔥 Merged TB Region on X-ray in 3D",
        margin=dict(l=10, r=10, b=10, t=40),
        scene=dict(zaxis=dict(title='Activation Intensity')),
        width=900
    )
    st.plotly_chart(fig_overlay, use_container_width=True)

else:
    st.info("Upload a chest X-ray to explore TB regions in 3D with heatmap overlay.")
