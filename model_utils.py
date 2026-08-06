from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import tensorflow as tf
from PIL import Image
from matplotlib import colormaps


IMAGE_SIZE: Tuple[int, int] = (224, 224)
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]


def load_trained_model(model_path: Path) -> tf.keras.Model:
    """Load a saved Keras model from disk."""
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}\n"
            "Place best_brain_tumor_model.keras inside the models folder."
        )
    return tf.keras.models.load_model(model_path)


def prepare_image(image: Image.Image) -> tuple[np.ndarray, Image.Image]:
    """Convert an uploaded image into a 224x224 RGB batch."""
    rgb_image = image.convert("RGB")
    resized_image = rgb_image.resize(IMAGE_SIZE)
    image_array = np.asarray(resized_image, dtype=np.float32)
    image_batch = np.expand_dims(image_array, axis=0)
    return image_batch, resized_image


def predict_image(
    model: tf.keras.Model,
    image_batch: np.ndarray,
) -> tuple[str, float, np.ndarray]:
    """Return predicted class, confidence, and all class probabilities."""
    probabilities = model.predict(image_batch, verbose=0)[0]
    predicted_index = int(np.argmax(probabilities))
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = float(probabilities[predicted_index])
    return predicted_class, confidence, probabilities


def _find_nested_feature_model(model: tf.keras.Model) -> tf.keras.Model:
    """Find the nested CNN backbone, such as EfficientNetB0."""
    candidates = [
        layer for layer in model.layers
        if isinstance(layer, tf.keras.Model)
    ]

    if not candidates:
        raise ValueError("No nested CNN feature model was found.")

    # Prefer the largest nested model, which is usually EfficientNet.
    return max(candidates, key=lambda layer: len(layer.layers))


def _find_last_conv_layer(feature_model: tf.keras.Model) -> tf.keras.layers.Layer:
    """Find the last layer in the backbone that produces a 4D feature map."""
    for layer in reversed(feature_model.layers):
        try:
            output_shape = layer.output.shape
        except Exception:
            continue

        if len(output_shape) == 4:
            return layer

    raise ValueError("No convolutional feature layer was found.")


def make_gradcam_heatmap(
    image_batch: np.ndarray,
    model: tf.keras.Model,
) -> np.ndarray:
    """
    Create a Grad-CAM heatmap for the model's predicted class.

    This shows which image regions influenced the prediction. It is not an
    exact tumour boundary and should not be interpreted as a diagnosis.
    """
    feature_model = _find_nested_feature_model(model)
    last_conv_layer = _find_last_conv_layer(feature_model)

    # Model from the backbone input to the selected feature map.
    backbone_to_conv = tf.keras.Model(
        feature_model.inputs,
        last_conv_layer.output,
    )

    # Recreate the classifier path after the backbone.
    backbone_layer_index = model.layers.index(feature_model)
    classifier_layers = model.layers[backbone_layer_index + 1:]

    with tf.GradientTape() as tape:
        feature_maps = backbone_to_conv(image_batch, training=False)
        tape.watch(feature_maps)

        x = feature_maps
        # Continue through any remaining backbone layers after last_conv_layer.
        last_conv_index = feature_model.layers.index(last_conv_layer)
        for layer in feature_model.layers[last_conv_index + 1:]:
            x = layer(x, training=False)

        # Continue through the top-level classifier layers.
        for layer in classifier_layers:
            try:
                x = layer(x, training=False)
            except TypeError:
                x = layer(x)

        predictions = x
        predicted_index = tf.argmax(predictions[0])
        predicted_score = predictions[:, predicted_index]

    gradients = tape.gradient(predicted_score, feature_maps)
    if gradients is None:
        raise ValueError("Grad-CAM gradients could not be calculated.")

    pooled_gradients = tf.reduce_mean(gradients, axis=(0, 1, 2))
    heatmap = tf.reduce_sum(feature_maps[0] * pooled_gradients, axis=-1)
    heatmap = tf.maximum(heatmap, 0)

    max_value = tf.reduce_max(heatmap)
    heatmap = tf.where(max_value > 0, heatmap / max_value, heatmap)

    return heatmap.numpy()


def overlay_gradcam(
    original_image: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.42,
) -> Image.Image:
    """Overlay a Grad-CAM heatmap onto the MRI image."""
    heatmap_uint8 = np.uint8(255 * heatmap)

    cmap = colormaps["jet"]
    coloured = cmap(np.arange(256))[:, :3]
    coloured_heatmap = coloured[heatmap_uint8]
    coloured_heatmap = Image.fromarray(
        np.uint8(coloured_heatmap * 255)
    ).resize(original_image.size)

    original_array = np.asarray(
        original_image.convert("RGB"),
        dtype=np.float32,
    )
    heatmap_array = np.asarray(coloured_heatmap, dtype=np.float32)

    overlay = (
        original_array * (1.0 - alpha)
        + heatmap_array * alpha
    )
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    return Image.fromarray(overlay)
