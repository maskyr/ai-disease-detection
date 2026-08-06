from pathlib import Path

from PIL import Image

from model_utils import (
    CLASS_NAMES,
    load_trained_model,
    predict_image,
    prepare_image,
)


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "best_brain_tumor_model.keras"


def main() -> None:
    model = load_trained_model(MODEL_PATH)

    image_path_text = input(
        "Enter the full path to an MRI image: "
    ).strip().strip('"')

    image_path = Path(image_path_text)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = Image.open(image_path)
    image_batch, _ = prepare_image(image)

    predicted_class, confidence, probabilities = predict_image(
        model,
        image_batch,
    )

    print("\nPrediction result")
    print("-" * 32)
    print(f"Predicted class: {predicted_class}")
    print(f"Confidence: {confidence * 100:.2f}%")
    print("\nClass probabilities:")

    for class_name, probability in zip(CLASS_NAMES, probabilities):
        print(f"{class_name}: {probability * 100:.2f}%")


if __name__ == "__main__":
    main()
