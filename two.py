import streamlit as st
import cv2
import numpy as np
import plotly.graph_objs as go
from PIL import Image

st.set_page_config(layout="wide")

st.title("🩻 3D X-ray TB Visualizer")

uploaded_file = st.file_uploader("📤 Upload Chest X-ray Image", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    # Read image using PIL then convert to OpenCV
    image = Image.open(uploaded_file).convert("L")
    img = np.array(image)
    img = cv2.resize(img, (256, 256))

    st.image(img, caption='Uploaded X-ray', use_column_width=True)

    z = img.astype(np.float32)
    x = np.linspace(0, 1, z.shape[1])
    y = np.linspace(0, 1, z.shape[0])
    x, y = np.meshgrid(x, y)

    fig = go.Figure(data=[go.Surface(z=z, x=x, y=y, colorscale='Gray')])
    fig.update_layout(
        title='🧠 Rotate to Explore TB Region',
        scene=dict(
            zaxis=dict(title='Intensity', range=[0, 255]),
            xaxis=dict(title='Width'),
            yaxis=dict(title='Height'),
        ),
        width=900,
        margin=dict(r=20, l=20, b=20, t=40),
    )

    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Upload a chest X-ray to begin.")
