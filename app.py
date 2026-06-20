# app.py
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import tensorflow as tf
from tensorflow import keras
import cv2
import os

st.set_page_config(
    page_title="Digit Recognizer",
    page_icon="🧠",
    layout="centered"
)

# ✅ Fix — absolute path, always finds the file
import os
MODEL_PATH = os.path.join(os.path.dirname(__file__), "digit_model.keras")
# ── Load Model ───────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model...")
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(
            f"❌ Model not found.\n\n"
            "Run  `python main.py`  first to train and save the model."
        )
        st.stop()
    return keras.models.load_model(MODEL_PATH)

model = load_model()

# ── Preprocessing ────────────────────────────────────────────────
def preprocess_canvas(image_data: np.ndarray):
    """
    280×280 RGBA canvas → 28×28×1 float32, MNIST-compatible.
    Returns: (tensor, debug_img) or (None, None) if canvas empty.
    """
    # Strategy 1: use alpha channel (cleanest stroke signal)
    img = image_data[:, :, 3].astype(np.float32)

    if img.max() < 10:
        return None, None

    # Strategy 4: blur to match MNIST smooth edges
    img = cv2.GaussianBlur(img, (7, 7), sigmaX=2)

    # Threshold to clean edges
    _, img = cv2.threshold(img, 30, 255, cv2.THRESH_BINARY)

    # Strategy 4: normalize stroke thickness with dilation
    kernel = np.ones((2, 2), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)

    # Tight crop around digit
    coords = cv2.findNonZero(img)
    if coords is None:
        return None, None
    x, y, w, h = cv2.boundingRect(coords)
    img = img[y:y+h, x:x+w]

    # Strategy 3: preserve aspect ratio — pad to square first
    size = max(w, h)
    padded = np.zeros((size, size), dtype=np.float32)
    y_off = (size - h) // 2
    x_off = (size - w) // 2
    padded[y_off:y_off+h, x_off:x_off+w] = img

    # Strategy 2: resize to 20×20, add 4px border = 28×28 (MNIST centering)
    small = cv2.resize(padded, (20, 20), interpolation=cv2.INTER_AREA)
    final = np.pad(small, 4, mode='constant', constant_values=0)

    # Save debug image BEFORE normalizing
    debug_img = final.astype(np.uint8)

    # Normalize → [0,1], shape → (1,28,28,1)
    final = final / 255.0
    tensor = final.reshape(1, 28, 28, 1).astype(np.float32)

    return tensor, debug_img

# ── UI ───────────────────────────────────────────────────────────
st.title("🧠 Digit Recognizer")
st.caption("Draw a digit (0–9) and click **Predict**")

col1, col2 = st.columns([1.2, 1])

with col1:
    st.markdown("#### ✏️ Draw here")
    canvas = st_canvas(
        fill_color="black",
        stroke_width=18,        # ~MNIST stroke thickness
        stroke_color="white",
        background_color="black",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas",
    )

with col2:
    st.markdown("#### 🔎 Result")

    predict_btn = st.button("Predict", use_container_width=True, type="primary")
    clear_btn   = st.button("Clear",   use_container_width=True)

    if clear_btn:
        st.rerun()

    if predict_btn:
        if canvas.image_data is None:
            st.warning("Draw something first!")
        else:
            tensor, debug_img = preprocess_canvas(canvas.image_data)

            if tensor is None:
                st.warning("Canvas is empty — draw a digit first.")
            else:
                probs  = model.predict(tensor, verbose=0)[0]   # (10,)
                pred   = int(np.argmax(probs))
                conf   = float(probs[pred]) * 100

                # ── Main result ──────────────────────────────
                st.metric("Predicted Digit", str(pred),
                          delta=f"{conf:.1f}% confidence")

                st.markdown("---")

                # ── Top-3 breakdown ──────────────────────────
                st.markdown("**Top 3 guesses**")
                top3 = np.argsort(probs)[::-1][:3]
                for idx in top3:
                    p = float(probs[idx])
                    st.markdown(f"`{idx}` — {p*100:.1f}%")
                    st.progress(p)

                st.markdown("---")

                # ── Strategy 5: Debug viewer ─────────────────
                with st.expander("🔬 What the model sees (28×28)"):
                    st.image(
                        debug_img,
                        caption="Preprocessed input sent to model",
                        width=140
                    )
                    st.caption(
                        "If this looks wrong (inverted, stretched, empty), "
                        "the preprocessing is the issue — not the model."
                    )
                    



