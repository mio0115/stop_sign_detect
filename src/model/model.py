import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, MaxPool2D
from tensorflow.keras import Sequential


class MyModel(tf.keras.Model):
    def __init__(self, input_shape: tuple[int] = (720, 1280, 3)):
        super(MyModel, self).__init__()

        self.input_shape = input_shape

        self.conv_layers = [
            Conv2D(filters=32, kernel_size=(3, 3), padding="same"),
            Conv2D(filters=32, kernel_size=(3, 3), padding="same"),
            MaxPool2D(pool_size=(2, 2), strides=2, padding="same"),
            Conv2D(filters=64, kernel_size=(3, 3), padding="same"),
            MaxPool2D(pool_size=(2, 2), strides=2, padding="same"),
            Conv2D(filters=128, kernel_size=(3, 3), padding="same"),
            Conv2D(filters=128, kernel_size=(3, 3), padding="same"),
            MaxPool2D(pool_size=(2, 2), strides=2, padding="same"),
        ]
        self.cls_ffn = Sequential([Dense(units=128), Dense(units=64), Dense(units=32)])
        self.head = Dense(units=1, activation="sigmoid")

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]

        x = inputs

        for idx in range(3):
            x = self.conv_layers[idx](x)

        x = tf.reshape(x, shape=(batch_size, 180 * 320 * 128))
        x = self.cls_ffn(x)
        # x = x + tmp_x

        output = self.head(x)

        return output


def build_model(lr=0.0001):
    model = MyModel()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=tf.keras.losses.MeanSquaredError(),
    )

    return model


if __name__ == "__main__":
    model = MyModel()

    print("hello, world")
