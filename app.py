from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

from model_utils import (
    CLASS_NAMES,
    load_trained_model,
    make_gradcam_heatmap,
    overlay_gradcam,
    predict_image,
    prepare_image,
)


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "best_brain_tumor_model.keras"


st.set_page_config(
    page_title="Brain MRI Classifier",
    layout="wide",
)


@st.cache_resource
def get_model():
    return load_trained_model(MODEL_PATH)


st.title("Brain MRI Tumour Classifier")

st.write(
    "Upload a brain MRI image to classify it as glioma, meningioma, "
    "pituitary tumour, or no tumour."
)


try:
    model = get_model()
except Exception as error:
    st.error(str(error))
    st.stop()

uploaded_file = st.file_uploader(
    "Upload an MRI image",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is not None:
    try:
        original_image = Image.open(uploaded_file).convert("RGB")
        image_batch, resized_image = prepare_image(original_image)

        predicted_class, confidence, probabilities = predict_image(
            model,
            image_batch,
        )

        st.subheader("Prediction")

        result_col, confidence_col = st.columns(2)

        with result_col:
            st.metric("Predicted class", predicted_class.title())

        with confidence_col:
            st.metric("Confidence", f"{confidence * 100:.2f}%")

        probability_frame = pd.DataFrame(
            {
                "Class": [name.title() for name in CLASS_NAMES],
                "Probability": probabilities,
            }
        )

        st.subheader("Class probabilities")
        st.bar_chart(probability_frame.set_index("Class"))

        display_frame = probability_frame.copy()
        display_frame["Probability"] = display_frame["Probability"].map(
            lambda value: f"{value * 100:.2f}%"
        )
        st.dataframe(
            display_frame,
            hide_index=True,
            use_container_width=True,
        )

        st.subheader("Model attention")

        original_col, gradcam_col = st.columns(2)

        with original_col:
            st.image(
                original_image,
                caption="Uploaded MRI",
                use_container_width=True,
            )

        try:
            heatmap = make_gradcam_heatmap(image_batch, model)
            gradcam_image = overlay_gradcam(
                resized_image,
                heatmap,
                alpha=0.42,
            )

            with gradcam_col:
                st.image(
                    gradcam_image,
                    caption="Grad-CAM attention overlay",
                    use_container_width=True,
                )

            st.info(
                "The coloured region shows where the classifier focused. "
                "It is not a precise tumour outline or medically verified location."
            )

        except Exception as gradcam_error:
            with gradcam_col:
                st.warning(
                    "The prediction worked, but the Grad-CAM image could not "
                    "be generated for this saved model."
                )
            st.caption(f"Technical details: {gradcam_error}")

    except Exception as error:
        st.error(f"Could not process the image: {error}")
