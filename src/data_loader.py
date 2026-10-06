import tensorflow as tf  # noqa: I001

from src.config import (
    DATA_DIR,
    IMG_SIZE,
    BATCH_SIZE,
    VALIDATION_SPLIT,
    SEED
)


def load_datasets():

    train_dataset = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=VALIDATION_SPLIT,
        subset="training",
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )

    validation_dataset = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=VALIDATION_SPLIT,
        subset="validation",
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )

    class_names = train_dataset.class_names

    print("\nClasses:")
    for index, class_name in enumerate(class_names):
        print(index, "->", class_name)

    # Improve pipeline performance
    AUTOTUNE = tf.data.AUTOTUNE

    train_dataset = train_dataset.prefetch(
        buffer_size=AUTOTUNE
    )

    validation_dataset = validation_dataset.prefetch(
        buffer_size=AUTOTUNE
    )

    return train_dataset, validation_dataset, class_names