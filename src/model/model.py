import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Dropout
from tensorflow.keras import Sequential


class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()

        self.conv_layers = [
            Conv2D(filters=64, kernel_size=(3, 3), padding="same"),
            Conv2D(filters=64, kernel_size=(3, 3), padding="same"),
            Conv2D(filters=32, kernel_size=(3, 3), padding="same"),
        ]
        self.cls_ffn = Sequential([Dense(units=32), Dense(units=32), Dense(units=16)])
        self.head = Dense(units=1, activation="sigmoid")

    def call(self, inputs):
        batch_size = tf.shape(x)[0]

        x = inputs

        for idx in range(3):
            x = self.conv_layers[idx](x)

        x = tf.reshape(x, shape=(batch_size, -1))
        tmp_x = self.cls_ffn(x)
        x = x + tmp_x

        output = self.head(x)

        return output


def build_model():
    model = MyModel()

    return model


if __name__ == "__main__":
    model = MyModel()

    print("hello, world")
