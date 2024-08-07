import os
from functools import partial

import tensorflow as tf
import cv2
import numpy as np


def _create_tf_example(img, label):
    norm_img = np.clip(img, 0.0, 255.0) / 255

    # breakpoint()

    tf_example = tf.train.Example(
        features=tf.train.Features(
            feature={
                "image/image": tf.train.Feature(
                    float_list=tf.train.FloatList(value=norm_img.flatten())
                ),
                "image/label": tf.train.Feature(
                    float_list=tf.train.FloatList(
                        value=[
                            label,
                        ]
                    )
                ),
            }
        )
    )

    return tf_example.SerializeToString()


def prep_image(
    path_to_dataset: str | None = None,
    aug_flip: bool = False,
    aug_brightness: bool = False,
    aug_noise: bool = False,
    scale_factor: int = 1,
):
    path = {
        "stop": os.path.join(path_to_dataset, "stop"),
        "not_stop": os.path.join(path_to_dataset, "not_stop"),
    }

    if aug_flip or aug_brightness or aug_noise:
        path_to_tfrecords = os.path.join(path_to_dataset, "tfrecords")

        if aug_flip:
            path_to_flip = os.path.join(path_to_tfrecords, "flip")
        if aug_brightness:
            path_to_brightness = os.path.join(path_to_tfrecords, "brightness")
        if aug_noise:
            path_to_noise = os.path.join(path_to_tfrecords, "noise")

    origin_wrt = tf.io.TFRecordWriter(
        os.path.join(path_to_tfrecords, "origin_img.tfrecord")
    )
    if aug_flip:
        flip_wrt = tf.io.TFRecordWriter(os.path.join(path_to_flip, "sample.tfrecord"))
    if aug_brightness:
        brightness_wrt = tf.io.TFRecordWriter(
            os.path.join(path_to_brightness, "sample.tfrecord")
        )
    if aug_noise:
        noise_wrt = tf.io.TFRecordWriter(os.path.join(path_to_noise, "sample.tfrecord"))
    for id_ in ["stop", "not_stop"]:
        path_to_images = path[id_]
        label = 0.0 if id_ == "not_stop" else 1.0

        for image_name in os.listdir(path_to_images):
            if not (image_name.endswith(".jpg") or image_name.endswith(".JPG")):
                continue

            img_full_path = os.path.join(path_to_images, image_name)

            img = cv2.imread(img_full_path)
            img = cv2.resize(img, (1280, 720))

            example = _create_tf_example(img=img, label=label)
            origin_wrt.write(example)

            if aug_flip:
                pass
            if aug_brightness:
                for _ in range(scale_factor):
                    new_img = img + np.random.normal(loc=0.0, scale=25.0, size=(1,))

                    example = _create_tf_example(img=new_img, label=label)

                    brightness_wrt.write(example)
            if aug_noise:
                for _ in range(scale_factor):
                    noise = np.random.normal(loc=0.0, scale=15, size=(720, 1280, 3))

                    new_img = img + noise
                    example = _create_tf_example(img=new_img, label=label)

                    noise_wrt.write(example)

    origin_wrt.close()
    if aug_flip:
        flip_wrt.close()
    if aug_brightness:
        brightness_wrt.close()
    if aug_noise:
        noise_wrt.close()


def load_data(
    path_to_tfrecord="/home/daniel/cv_project/stop_sign_detect/dataset/tfrecords",
    parse_function=None,
    feature_description=None,
    batch_size=8,
):
    tfrecord_files = [
        os.path.join(path_to_tfrecord, "brightness", "sample.tfrecord"),
        os.path.join(path_to_tfrecord, "noise", "sample.tfrecord"),
        os.path.join(path_to_tfrecord, "origin_img.tfrecord"),
    ]

    parse_fn = partial(parse_function, ft_desc=feature_description)
    raw_dataset = tf.data.TFRecordDataset(tfrecord_files)
    parsed_dataset = raw_dataset.map(parse_fn)

    return parsed_dataset.batch(batch_size=batch_size)


def parse_stop_sign(proto, ft_desc):
    parsed_ft = tf.io.parse_single_example(proto, ft_desc)
    img = tf.reshape(parsed_ft["image/image"], (720, 1280, 3))
    label = parsed_ft["image/label"]

    return img, label


if __name__ == "__main__":
    # prep_image(
    #    path_to_dataset=os.path.join(
    #        "/home", "daniel", "cv_project", "stop_sign_detect", "dataset"
    #    ),
    #    aug_brightness=True,
    #    aug_noise=True,
    #    scale_factor=10,
    # )

    ft_desc = {
        "image/image": tf.io.FixedLenFeature([1280 * 720 * 3], dtype=tf.float32),
        "image/label": tf.io.FixedLenFeature([1], dtype=tf.float32),
    }
    dataset = load_data(parse_function=parse_stop_sign, feature_description=ft_desc)

    for batch in dataset:
        print(batch)
        break
