import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2

# Custom CSS for styling
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap');

* {
    font-family: 'Poppins', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    color: #ffffff;
}

.stButton>button {
    background: linear-gradient(90deg, #2dd4bf 0%, #3b82f6 100%);
    color: #1a1a2e;
    border: none;
    padding: 0.8rem 2rem;
    border-radius: 15px;
    font-weight: 600;
    transition: 0.3s;
}

.stButton>button:hover {
    transform: scale(1.05);
    box-shadow: 0 4px 15px rgba(45, 212, 191, 0.4);
}

.stImage img {
    border-radius: 20px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    transition: transform 0.3s ease;
    border: 2px solid rgba(45, 212, 191, 0.3);
}

.stImage:hover img {
    transform: scale(1.02);
}

.prediction-card {
    background: rgba(255, 255, 255, 0.1);
    padding: 1.5rem;
    border-radius: 15px;
    margin: 1rem 0;
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.2);
    animation: fadeIn 0.5s ease-in;
}

.confidence-bar {
    height: 10px;
    background: linear-gradient(90deg, #2dd4bf 0%, #3b82f6 100%);
    border-radius: 5px;
    margin: 0.5rem 0;
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}

.title-gradient {
    background: linear-gradient(45deg, #2dd4bf, #3b82f6);
    -webkit-background-clip: text;
    background-clip: text;
    color: white !important;
    font-weight: 700 !important;
}

.social-links {
    display: flex;
    gap: 1rem;
    justify-content: center;
    margin-top: 1rem;
}

.social-links a {
    color: #2dd4bf !important;
    text-decoration: none;
    transition: 0.3s;
    padding: 0.5rem 1rem;
    border-radius: 8px;
    border: 1px solid rgba(45, 212, 191, 0.3);
}

.social-links a:hover {
    color: #3b82f6 !important;
    transform: translateY(-2px);
    background: rgba(45, 212, 191, 0.1);
}

.model-badge {
    background: rgba(45, 212, 191, 0.2);
    padding: 0.5rem 1rem;
    border-radius: 8px;
    border: 1px solid #2dd4bf;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

# App Header
st.markdown("<h1 class='title-gradient' style='text-align: center; text-shadow: 2px 2px 4px rgba(0,0,0,0.5);'>🐯 WildVision Pro</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; margin-bottom: 2rem; color: #2dd4bf;'>Advanced Animal Recognition with 96% Accuracy</h3>", unsafe_allow_html=True)

# Class names
class_names = [
    "Bear", "Bird", "Cat", "Cow", "Deer", 
    "Dog", "Dolphin", "Elephant", "Giraffe", 
    "Horse", "Kangaroo", "Lion", "Panda", 
    "Tiger", "Zebra"
]

# Model loading
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('final_model.keras')

model = load_model()

# Image preprocessing for VGG19
def preprocess_image(image):
    img = Image.open(image).convert('RGB')
    img = np.array(img)
    img = cv2.resize(img, (224, 224))  # VGG19 requires 224x224
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Sidebar with model info
with st.sidebar:
    st.markdown("## 🚀 Model Architecture")
    st.markdown("<div class='model-badge'>Xception Backbone</div>", unsafe_allow_html=True)
    st.markdown("**Key Features:**")
    st.markdown("- Global Average Pooling 2D")
    st.markdown("- 512-neuron Dense Layer")
    st.markdown("- Dropout Regularization")
    st.markdown("- 15-class Output")
    st.markdown("**Test Accuracy:** 96%")

# Main content
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("### 📤 Upload Animal Image")
    uploaded_file = st.file_uploader("", type=['jpg', 'jpeg', 'png'], 
                                   label_visibility='collapsed')

    if uploaded_file:
        st.markdown("### Image Preview")
        image = Image.open(uploaded_file)
        st.image(image, use_container_width=True, caption="Uploaded Image", 
                output_format='auto', clamp=True, channels='RGB')

with col2:
    if uploaded_file:
        st.markdown("### 🔍 Advanced Analysis")
        if st.button("Analyze Image 🚀", use_container_width=True):
            with st.spinner("Processing with AI..."):
                # Preprocess and predict
                processed_img = preprocess_image(uploaded_file)
                predictions = model.predict(processed_img)[0]
                top3_indices = np.argsort(predictions)[-3:][::-1]
                
                # Display results
                st.balloons()
                for i in top3_indices:
                    confidence = predictions[i]
                    with st.container():
                        st.markdown(f"<div class='prediction-card'>"
                                    f"<h4>{class_names[i]}</h4>"
                                    f"<div class='confidence-bar' style='width: {confidence*100:.1f}%'></div>"
                                    f"<p>Confidence: {confidence*100:.2f}%</p>"
                                    "</div>", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #a8a8a8; margin-top: 2rem;'>
    <div class='social-links'>
        <a href='https://github.com/ankitparwatkar' target='_blank'>GitHub</a>
        <a href='https://linkedin.com/in/ankitparwatkar' target='_blank'>LinkedIn</a>
    </div>
    <div style='margin-top: 1.5rem; font-size: 0.9rem;'>
        🐯 Developed by <strong>Ankit Parwatkar</strong><br>
        © 2025 All Rights Reserved<br>
        🌍 Ethical AI for Wildlife Conservation
    </div>
</div>
""", unsafe_allow_html=True)