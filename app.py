import streamlit as st
import numpy as np
import importlib
from PIL import Image
from pathlib import Path

try:
    tf = importlib.import_module('tensorflow')
    keras = importlib.import_module('keras')
except ImportError:
    tf = None
    keras = None


PROJECT_DIR = Path(__file__).parent
MODEL_PATH = PROJECT_DIR / 'model.keras'
LOGO_PATH = PROJECT_DIR / 'logo.png'
NORMAL_IMAGE_PATH = PROJECT_DIR / 'Normal.jpg'
PNEUMONIA_IMAGE_PATH = PROJECT_DIR / 'Pneumonia.jpg'

# Load Model
model = None
if keras and MODEL_PATH.is_file() and MODEL_PATH.stat().st_size > 0:
    try:
        model = keras.models.load_model(MODEL_PATH)
    except (OSError, ValueError):
        model = None

# Interface

# title
with st.container():
    image_col, heading_col = st.columns([.5, 1.5], vertical_alignment='center')
    
    with image_col:
        st.image(LOGO_PATH, width=150)
        
    with heading_col:
        st.write('# 🖼️:rainbow[Pneumonia Detector]👩‍⚕️')
        st.write('**:blue[Pneumonia Detector is a deep learning image classification model, that classifies the l...')

# Input and Output
with st.container():
    image = st.file_uploader('**:orange[UPLOAD XRAY IMAGE]**', accept_multiple_files=False)
    if image:
        input_col, output_col = st.columns(2, border=False)
        
        with input_col:
            st.image(image=image, width='stretch')
            
        with output_col:
            if not MODEL_PATH.is_file() or MODEL_PATH.stat().st_size == 0:
                st.error('Prediction is unavailable because model.keras is empty or missing. Run the final training cell in Pneumonia_Detector.ipynb to create it.')
            elif model is None or tf is None:
                st.error('Prediction is unavailable because TensorFlow/Keras is not installed. Run: py -m pip install tensorflow keras')
            else:
                image = Image.open(image).convert('RGB').resize((180, 180))
                image_array = np.asarray(image, dtype=np.float32)
                prediction = model.predict(tf.expand_dims(image_array, axis=0), verbose=0)
                if float(np.asarray(prediction).squeeze()) < 0.5:
                    st.image(NORMAL_IMAGE_PATH)
                else:
                    st.image(PNEUMONIA_IMAGE_PATH)

# Share Contents
with st.container():
    youtube_col, linkedin_col, huggingface_col = st.columns(3, border=False)
    
    with youtube_col:
        st.link_button('**YOUTUBE**', icon=':material/play_circle:', url='', width='stretch', help='Code...')
        
    with linkedin_col:
        st.link_button('**LINKEDIN**', icon=':material/group_add:', url='https://www.linkedin.com/in/madhavpani/')
        
    with huggingface_col:
        st.link_button('**HUGGING FACE**', icon=':material/deployed_code:', url='', width='stretch', help='...')
