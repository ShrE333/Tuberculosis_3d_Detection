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
st.title("🧠 TB X-ray 3D Visualizer with Grad-CAM")

# Load model
@st.cache_resource
def load_tb_model():
    return load_model("tb_model.h5")

model = load_tb_model()
gradcam = Gradcam(model, model_modifier=ReplaceToLinear())
score = CategoricalScore([1])  # Class 1 = TB

# Image preprocessing
def preprocess(image):
    image = cv2.resize(image, (224, 224))
    image = image.astype(np.float32) / 255.
    return np.expand_dims(image, axis=0)

uploaded_file = st.file_uploader("📤 Upload Chest X-ray Image", type=['jpg', 'jpeg', 'png'])

if uploaded_file:
    # Load & show image
    pil_img = Image.open(uploaded_file).convert("RGB")
    np_img = np.array(pil_img)
    st.image(pil_img, caption="🩻 Uploaded X-ray", use_column_width=True)

    # Preprocess for prediction
    input_tensor = preprocess(np_img)

    # Predict
    preds = model.predict(input_tensor)
    class_names = ['Normal', 'TB']
    predicted_class = class_names[np.argmax(preds)]
    st.success(f"🔍 Prediction: **{predicted_class}** ({np.max(preds)*100:.2f}%)")

    # Generate Grad-CAM
    cam = gradcam(score, input_tensor, penultimate_layer=-1)
    heatmap = np.uint8(255 * cam[0])  # 0-255
    heatmap_resized = cv2.resize(heatmap, (256, 256))

    # Resize X-ray to match for 3D
    gray_img = cv2.resize(cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY), (256, 256))
    gray_img = gray_img.astype(np.float32)

    # 3D axes
    x = np.linspace(0, 1, 256)
    y = np.linspace(0, 1, 256)
    x, y = np.meshgrid(x, y)

    # --- 3D Visualizations ---
    col1, col2 = st.columns(2)

    # Full X-ray with heatmap overlay
    with col1:
        full_surface = gray_img
        fig_full = go.Figure(data=[go.Surface(z=full_surface, x=x, y=y, colorscale='Gray')])
        fig_full.update_layout(title="🩻 Full X-ray in 3D", margin=dict(l=0, r=0, b=0, t=30), width=500)
        st.plotly_chart(fig_full, use_container_width=True)

    # TB-only heatmap as surface
    with col2:
        tb_surface = heatmap_resized.astype(np.float32)
        fig_tb = go.Figure(data=[go.Surface(z=tb_surface, x=x, y=y, colorscale='Hot')])
        fig_tb.update_layout(title="🔥 TB Activation Region in 3D", margin=dict(l=0, r=0, b=0, t=30), width=500)
        st.plotly_chart(fig_tb, use_container_width=True)

else:
    st.info("Upload a chest X-ray to start visualizing in 3D with TB regions highlighted.")
