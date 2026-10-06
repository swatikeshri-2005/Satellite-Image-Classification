import numpy as np  # noqa: I001
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from sklearn.metrics import (
    classification_report,
    confusion_matrix
)

from src.config import MODEL_PATH, OUTPUT_DIR
from src.data_loader import load_datasets


def main():

    print("Loading dataset...")

    _, validation_dataset, class_names = load_datasets()

    print("\nLoading trained model...")

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    print("\nGenerating predictions...")

    y_true = []
    y_pred = []

    for images, labels in validation_dataset:

        predictions = model.predict(
            images,
            verbose=0
        )

        predictions = np.argmax(
            predictions,
            axis=1
        )

        y_true.extend(
            labels.numpy()
        )

        y_pred.extend(
            predictions
        )

    print("\n==============================")
    print("CLASSIFICATION REPORT")
    print("==============================\n")

    print(
        classification_report(
            y_true,
            y_pred,
            target_names=class_names
        )
    )

    # Create confusion matrix
    matrix = confusion_matrix(
        y_true,
        y_pred
    )

    print("\nCreating confusion matrix...")

    plt.figure(
        figsize=(12, 10)
    )

    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        xticklabels=class_names,
        yticklabels=class_names
    )

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(
        "Satellite Image Classification - Confusion Matrix"
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = OUTPUT_DIR / "confusion_matrix.png"

    plt.tight_layout()

    plt.savefig(
        output_path
    )

    plt.close()

    print(
        f"\nConfusion matrix saved to:"  # noqa: F541
    )

    print(output_path)

    print("\nEvaluation completed successfully!")


if __name__ == "__main__":
    main()