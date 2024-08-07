import tensorflow as tf

from .model.model import build_model
from .data_prep import load_data, parse_stop_sign


def train(model: tf.keras.Model = None, dataset=None):
    history = model.fit(
        dataset,
        batch_size=8,
        epochs=10,
    )

    return history


def infer(model, image):
    return model(image[tf.newaxis, ...])[0]


if __name__ == "__main__":
    ft_desc = {
        "image/image": tf.io.FixedLenFeature([1280 * 720 * 3], dtype=tf.float32),
        "image/label": tf.io.FixedLenFeature([1], dtype=tf.float32),
    }

    model = build_model()
    dataset = load_data(parse_function=parse_stop_sign, feature_description=ft_desc)

    history = train(model=model, dataset=dataset)
