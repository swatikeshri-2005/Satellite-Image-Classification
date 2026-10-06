import numpy as np  # noqa: I001
import tensorflow as tf
import streamlit as st
from PIL import Image


# ==========================================
# CONFIGURATION
# ==========================================

MODEL_PATH = "models/satellite_classifier.keras"

CLASS_NAMES = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake"
]


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Satellite Image Classifier",
    page_icon="🛰️",
    layout="centered"
)


# ==========================================
# TITLE
# ==========================================

st.title("🛰️ Satellite Image Classification")

st.write(
    "Upload a satellite image and our deep learning "
    "model will predict its land-cover category."
)

st.divider()


# ==========================================
# LOAD MODEL
# ==========================================

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model


# ==========================================
# FILE UPLOAD
# ==========================================

uploaded_file = st.file_uploader(
    "Upload a satellite image",
    type=["jpg", "jpeg", "png"]
)


# ==========================================
# IMAGE + PREDICTION
# ==========================================

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Satellite Image",
        use_container_width=True
    )

    st.divider()

    if st.button("Classify Image", use_container_width=True):

        with st.spinner("Analyzing satellite image..."):

            # Load trained model
            model = load_model()

            # Resize image
            image_resized = image.resize((224, 224))

            # Convert to NumPy array
            image_array = np.array(image_resized)

            # Add batch dimension
            image_array = np.expand_dims(
                image_array,
                axis=0
            )

            # Make prediction
            predictions = model.predict(
                image_array,
                verbose=0
            )[0]

            # Find highest probability
            predicted_index = np.argmax(predictions)

            predicted_class = CLASS_NAMES[predicted_index]

            confidence = predictions[predicted_index] * 100


        # ======================================
        # DISPLAY RESULT
        # ======================================

        st.success(
            f"Prediction: {predicted_class}"
        )

        st.metric(
            "Confidence",
            f"{confidence:.2f}%"
        )

        st.divider()

        st.subheader("Class Probabilities")

        probabilities = {
            CLASS_NAMES[i]: float(predictions[i])
            for i in range(len(CLASS_NAMES))
        }

        st.bar_chart(probabilities)