import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Input, Dense, Lambda, Conv2D, MaxPooling2D, UpSampling2D, ReLU, BatchNormalization, concatenate
from tensorflow.keras.models import Model, Sequential


def tf_squared():
    x = Input(shape=(5,))
    y = tf.square(x)
    model = Model(x, y)

    return model

# model = Sequential([
#     Input(shape=(5,)),
#     Lambda(lambda x: tf.square(x))
# ])


def tf_logistic():
    x = Input(shape=(5,))
    y = Dense(16, activation='softmax')(x)

    # (None, 5) -> (None, 16)
    model = Model(x, y)
    return model

# model = Sequential()
# model.add(Input(shape=(5,)))
# model.add(Dense(32, activation='relu'))


def tf_deep_n_wide():
    input_a = Input(shape=(5,), name="wide_input")
    input_b = Input(shape=(6,), name="deep_input")
    hidden1 = Dense(30, activation="relu")(input_b)
    hidden2 = Dense(30, activation="relu")(hidden1)
    concat = concatenate([input_a, hidden2])
    output = Dense(1, name="output")(concat)
    model = Model(inputs=[input_a, input_b], outputs=[output])
    return model


class WideAndDeepModel(Model):
    def __init__(self, units=30, activation="relu", **kargs):
        super().__init__(**kargs)
        self.hidden1 = Dense(units, activation=activation)
        self.hidden2 = Dense(units, activation=activation)
        self.outputs = Dense(1)

    def call(self, inputs):
        input_a, input_b = inputs
        hidden1 = self.hidden1(input_b)
        hidden2 = self.hidden2(hidden1)
        concat = concatenate([input_a, hidden2])
        output = self.outputs(concat)
        return output

    def summary(self):
        input_a = Input(shape=(5,))
        input_b = Input(shape=(6,))
        Model(inputs=[input_a, input_b], outputs=self.call([input_a, input_b])).summary()


def main():
    arr1 = np.array([[1, 2, 3, 4, 5],
                     [6, 7, 8, 9, 10],
                     [11, 12, 13, 14, 15]], dtype=np.float64)
    arr2 = np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                     [1.1, 1.2, 1.3, 1.4, 1.5, 1.6],
                     [2.1, 2.2, 2.3, 2.4, 2.5, 2.6]], dtype=np.float64)
    # model = tf_deep_n_wide()

    model = WideAndDeepModel()
    # model.build(input_shape=[(None, 5), (None, 6)])
    model.summary()
    y = model.predict([arr1, arr2])
    print(y)


if __name__ == "__main__":
    main()
