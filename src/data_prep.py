import os

import tensorflow as tf
import cv2


def _create_tf_example(img, label):
    tf_example = tf.train.Example(
        features=tf.train.Features(
            feature={
                "image/image": tf.train.Feature(
                    float_list=tf.train.FloatList(value=img)
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

    return tf_example


def prep_image(path_to_dataset, aug_flip=False, aug_brightness=False, aug_noise=False):
    path_to_stop = os.path.join(path_to_dataset, "stop")
    path_to_no_stop = os.path.join(path_to_dataset, "not_stop")

    if aug_flip or aug_brightness or aug_noise:
        path_to_tfrecords = os.path.join(path_to_dataset, "tfrecords")

        if aug_flip:
            path_to_flip = os.path.join(path_to_tfrecords, "flip")
        if aug_brightness:
            path_to_brightness = os.path.join(path_to_tfrecords, "brightness")
        if aug_noise:
            path_to_noise = os.path.join(path_to_tfrecords, "noise")

    for image_name in os.listdir(path_to_stop):
        img_full_path = os.path.join(path_to_stop, image_name)

        img = cv2.imread(img_full_path)
        img = cv2.resize(img, (1280, 720))
