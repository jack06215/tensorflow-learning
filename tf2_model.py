import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Lambda, Conv2D, MaxPooling2D, UpSampling2D, ReLU, BatchNormalization
from tensorflow.keras.models import Model, Sequential

IMAGE_SIZE = (256, 256, 3)
N_CLASSES = 7


# def segnet(input_shape, n_classes):
#     input = Input(shape=input_shape)

#     # Encoder
#     x = Conv2D(64, 3, padding="same", kernel_initializer="he_normal")(input)
#     x = BatchNormalization()(x)
#     x = ReLU()(x)
#     x = Conv2D(64, 3, padding="same", kernel_initializer="he_normal")(x)
#     x = BatchNormalization()(x)
#     x = ReLU()(x)
#     x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)
#     p1 = x


#     output = Conv2D(n_classes, 1, activation="softmax")(x)
#     model = Model(inputs=input, outputs=output)

#     return model


# segnet_model = segnet(input_shape=IMAGE_SIZE, n_classes=N_CLASSES)
# print(segnet_model.summary())


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


def main():
    model = tf_logistic()
    print(model.summary())
    y = model.predict([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]])
    print(y)

if __name__ == "__main__":
    main()