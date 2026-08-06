from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)


PROJECT_ROOT = Path(__file__).resolve().parent
TEST_DIRECTORY = PROJECT_ROOT / "data" / "brain_tumor_mri" / "Testing"
MODEL_PATH = PROJECT_ROOT / "models" / "best_brain_tumor_model.keras"
RESULTS_DIRECTORY = PROJECT_ROOT / "results"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    if not TEST_DIRECTORY.exists():
        raise FileNotFoundError(
            f"Testing dataset not found: {TEST_DIRECTORY}"
        )

    RESULTS_DIRECTORY.mkdir(parents=True, exist_ok=True)

    test_dataset = tf.keras.utils.image_dataset_from_directory(
        TEST_DIRECTORY,
        shuffle=False,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
    )

    class_names = test_dataset.class_names
    test_dataset = test_dataset.prefetch(tf.data.AUTOTUNE)

    model = tf.keras.models.load_model(MODEL_PATH)

    test_loss, test_accuracy = model.evaluate(test_dataset, verbose=1)
    print(f"\nTest loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy * 100:.2f}%")

    probabilities = model.predict(test_dataset, verbose=1)
    predicted_labels = np.argmax(probabilities, axis=1)

    true_labels = np.concatenate(
        [
            np.argmax(labels.numpy(), axis=1)
            for _, labels in test_dataset
        ]
    )

    report = classification_report(
        true_labels,
        predicted_labels,
        target_names=class_names,
        digits=4,
    )

    print("\nClassification report:\n")
    print(report)

    report_path = RESULTS_DIRECTORY / "classification_report.txt"
    report_path.write_text(report, encoding="utf-8")

    matrix = confusion_matrix(true_labels, predicted_labels)
    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=class_names,
    )
    display.plot(xticks_rotation=45, values_format="d")

    plt.title("Brain Tumour Classification Confusion Matrix")
    plt.tight_layout()

    matrix_path = RESULTS_DIRECTORY / "confusion_matrix.png"
    plt.savefig(matrix_path, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Saved report: {report_path}")
    print(f"Saved confusion matrix: {matrix_path}")


if __name__ == "__main__":
    main()
