import os  # noqa: I001
import tensorflow as tf

from src.config import (
    MODEL_DIR,
    MODEL_PATH,
    EPOCHS
)

from src.data_loader import load_datasets
from src.model import create_model


def main():

    print("Loading dataset...")

    train_dataset, validation_dataset, class_names = (
        load_datasets()
    )

    print("\nCreating model...")

    model = create_model(
        num_classes=len(class_names)
    )

    model.summary()

    # Create model directory
    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    # Save best model
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        MODEL_PATH,
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1
    )

    # Stop if validation performance stops improving
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=4,
        restore_best_weights=True
    )

    print("\nStarting training...")

    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=EPOCHS,
        callbacks=[
            checkpoint,
            early_stopping
        ]
    )

    print("\nTraining completed.")

    print(
        f"Best model saved to: {MODEL_PATH}"
    )

    return history


if __name__ == "__main__":
    main()