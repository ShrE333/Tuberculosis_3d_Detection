import streamlit as st
import numpy as np
import cv2
from PIL import Image
import plotly.graph_objs as go
from tensorflow.keras.models import load_model
from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
from tf_keras_vis.utils.scores import CategoricalScore

st.set_page_config(layout="wide")
st.title("🧠 3D Grad-CAM TB Visualizer")

# Load model
@st.cache_resource
def load_tb_model():
    return load_model("tb_model.h5")

model = load_tb_model()
gradcam = Gradcam(model, model_modifier=ReplaceToLinear())
score = CategoricalScore([1])  # Class 1: TB

# Preprocess image
def preprocess(image):
    image = cv2.resize(image, (224, 224))
    image = image.astype(np.float32) / 255.
    return np.expand_dims(image, axis=0)

# Upload image
uploaded_file = st.file_uploader("📤 Upload Chest X-ray Image", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    # Convert to array
    pil_img = Image.open(uploaded_file).convert("RGB")
    original_np = np.array(pil_img)
    st.image(original_np, caption="🩻 Uploaded X-ray", use_column_width=True)

    # Grad-CAM
    input_tensor = preprocess(original_np)
    cam = gradcam(score, input_tensor, penultimate_layer=-1)
    heatmap = np.uint8(255 * cam[0])
    heatmap_resized = cv2.resize(heatmap, (256, 256))

    # 3D Visualization
    z = heatmap_resized.astype(np.float32)
    x = np.linspace(0, 1, z.shape[1])
    y = np.linspace(0, 1, z.shape[0])
    x, y = np.meshgrid(x, y)

    fig = go.Figure(data=[go.Surface(z=z, x=x, y=y, colorscale='Hot')])
    fig.update_layout(
        title='🔥 3D Grad-CAM: Explore TB Region',
        scene=dict(
            zaxis=dict(title='Activation', range=[0, 255]),
            xaxis=dict(title='Width'),
            yaxis=dict(title='Height'),
        ),
        width=900,
        margin=dict(r=20, l=20, b=20, t=40),
    )

    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Upload a chest X-ray to see 3D TB heatmap visualization.")
