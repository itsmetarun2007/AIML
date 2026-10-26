import os
import tensorflow as tf

MODEL_PATH = "mnist_cnn_model.keras"


def load_saved_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file '{MODEL_PATH}' was not found. "
            "Place the trained model in the project folder or train a new model."
        )

    print(f"Loading saved model from {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH)
    return model


if __name__ == "__main__":
    model = load_saved_model()
    model.summary()
