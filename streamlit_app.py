import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import os

st.set_page_config(page_title="Handwritten Digit Recognizer", page_icon="✍️", layout="centered")

st.markdown("""
<style>
.title    { font-size: 2.4rem; font-weight: 900; text-align: center; color: #a78bfa; }
.subtitle { text-align: center; color: #94a3b8; margin-bottom: 1.5rem; }
.pred-box { background: linear-gradient(135deg,#1e1b4b,#312e81); border: 2px solid #818cf8;
            border-radius: 12px; padding: 1.5rem; text-align: center;
            font-size: 5rem; font-weight: 900; color: #c7d2fe; margin: 1rem 0; }
.conf-label { color: #818cf8; font-size: 1rem; text-align: center; margin-top: -0.5rem; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">✍️ Handwritten Digit Recognizer</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload an image of a handwritten digit (0–9) and the CNN will classify it</div>', unsafe_allow_html=True)

ROOT = os.path.dirname(__file__)
MODEL_PATH = os.path.join(ROOT, "models_registry", "best_model.keras")

@st.cache_resource(show_spinner="Loading model…")
def load_model():
    try:
        import tensorflow as tf
        if os.path.exists(MODEL_PATH):
            return tf.keras.models.load_model(MODEL_PATH), None
        return None, "not_found"
    except Exception as e:
        return None, str(e)

def preprocess_image(img: Image.Image) -> np.ndarray:
    img = img.convert("L")          # grayscale
    img = ImageOps.invert(img)      # white digit on black bg → black bg on white
    img = img.resize((28, 28), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    # Auto-threshold: if image looks inverted after conversion, flip it
    if arr.mean() > 0.5:
        arr = 1.0 - arr
    return arr.reshape(1, 28, 28, 1)

model, model_err = load_model()

# ── Upload tab ────────────────────────────────────────────────────────────────
st.subheader("Upload an image")
uploaded = st.file_uploader(
    "Supports PNG, JPG — works best with dark background and light digit",
    type=["png", "jpg", "jpeg"]
)

if uploaded:
    img = Image.open(uploaded)
    col_img, col_result = st.columns([1, 1])

    with col_img:
        st.image(img, caption="Uploaded image", width=200)

    with col_result:
        if model_err == "not_found":
            st.warning("Model not trained yet.")
            st.code("python train.py", language="bash")
        elif model_err:
            st.error(f"Error loading model: {model_err}")
        else:
            arr = preprocess_image(img)
            preds = model.predict(arr, verbose=0)[0]
            digit = int(np.argmax(preds))
            conf  = float(preds[digit]) * 100
            st.markdown(f'<div class="pred-box">{digit}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="conf-label">Confidence: {conf:.1f}%</div>', unsafe_allow_html=True)
            st.markdown("")
            st.markdown("**All probabilities:**")
            for i, p in enumerate(preds):
                st.progress(float(p), text=f"Digit {i}: {p*100:.1f}%")

# ── Drawing tab ───────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Or draw a digit")

try:
    from streamlit_drawable_canvas import st_canvas

    canvas_result = st_canvas(
        fill_color="black",
        stroke_width=18,
        stroke_color="white",
        background_color="black",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas",
    )

    if st.button("Predict drawn digit", type="primary"):
        if canvas_result.image_data is not None:
            img_array = canvas_result.image_data.astype(np.uint8)
            img = Image.fromarray(img_array).convert("RGBA").convert("L")
            if model_err == "not_found":
                st.warning("Train the model first: `python train.py`")
            elif model_err:
                st.error(f"Error: {model_err}")
            else:
                img_resized = img.resize((28, 28), Image.LANCZOS)
                arr = np.array(img_resized, dtype=np.float32) / 255.0
                arr = arr.reshape(1, 28, 28, 1)
                preds = model.predict(arr, verbose=0)[0]
                digit = int(np.argmax(preds))
                conf  = float(preds[digit]) * 100
                st.markdown(f'<div class="pred-box">{digit}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="conf-label">Confidence: {conf:.1f}%</div>', unsafe_allow_html=True)
        else:
            st.info("Draw a digit on the canvas above first.")

except ImportError:
    st.info("Install `streamlit-drawable-canvas` to enable in-browser drawing:")
    st.code("pip install streamlit-drawable-canvas", language="bash")

st.markdown("---")
st.markdown("**About:** CNN trained on MNIST (60,000 images). Full MLOps pipeline with DVC data versioning, MLflow experiment tracking, Docker containerisation, and CI/CD.")
