import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, ReLU
from tensorflow.keras.layers import BatchNormalization, Dropout, Lambda, Conv2DTranspose
from tensorflow.keras import backend as K

def weighted_CCE_loss(class_weights):
    """ Weighted crossentropy loss """
    def weighted_loss(y_true, y_pred):
        class_weights_tensor = tf.cast(class_weights, tf.float32)
        y_pred_weighted = y_pred * class_weights_tensor
        cce_loss = - K.sum(y_true * K.log(y_pred_weighted))
        return cce_loss
    return weighted_loss


def multi_unet(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS, N_CLASSES):
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    KERNEL = tf.keras.initializers.HeNormal()

    # Encoder
    c1 = Conv2D(64, (3, 3), padding='same', kernel_initializer=KERNEL)(inputs)
    c1 = ReLU()(c1)
    c1 = BatchNormalization()(c1)
    c1 = Conv2D(64, (3, 3), padding='same', kernel_initializer=KERNEL)(c1)
    c1 = BatchNormalization()(c1)
    c1 = ReLU()(c1)
    p1 = MaxPooling2D((2, 2), strides=2)(c1)

    c2 = Conv2D(128, (3, 3), padding='same', kernel_initializer=KERNEL)(p1)
    c2 = ReLU()(c2)
    c2 = BatchNormalization()(c2)
    c2 = Conv2D(128, (3, 3), padding='same', kernel_initializer=KERNEL)(c2)
    c2 = BatchNormalization()(c2)
    c2 = ReLU()(c2)
    p2 = MaxPooling2D((2, 2), strides=2)(c2)

    c3 = Conv2D(256, (3, 3), padding='same', kernel_initializer=KERNEL)(p2)
    c3 = ReLU()(c3)
    c3 = BatchNormalization()(c3)
    c3 = Conv2D(256, (3, 3), padding='same', kernel_initializer=KERNEL)(c3)
    c3 = ReLU()(c3)
    c3 = BatchNormalization()(c3)
    p3 = MaxPooling2D((2, 2), strides=2)(c3)

    c4 = Conv2D(512, (3, 3), padding='same', kernel_initializer=KERNEL)(p3)
    c4 = ReLU()(c4)
    c4 = BatchNormalization()(c4)
    c4 = Conv2D(512, (3, 3), padding='same', kernel_initializer=KERNEL)(c4)
    c4 = ReLU()(c4)
    c4 = BatchNormalization()(c4)
    p4 = MaxPooling2D((2, 2), strides=2)(c4)

    c5 = Conv2D(1024, (3, 3), padding='same', kernel_initializer=KERNEL)(p4)
    c5 = ReLU()(c5)
    c5 = BatchNormalization()(c5)

    # Decoder
    u6 = Conv2DTranspose(512, (2, 2), strides=2, padding='same')(c5)
    u6 = tf.concat([c4, u6], axis=3)
    c6 = Conv2D(512, (3, 3), padding='same', kernel_initializer=KERNEL)(u6)
    c6 = ReLU()(c6)
    c6 = BatchNormalization()(c6)
    c6 = Conv2D(512, (3, 3), padding='same', kernel_initializer=KERNEL)(c6)
    c6 = ReLU()(c6)
    c6 = BatchNormalization()(c6)

    u7 = Conv2DTranspose(64, (2, 2), strides=2, padding='same')(c6)
    u7 = tf.concat([c3, u7], axis=3)
    c7 = Conv2D(256, (3, 3), padding='same', kernel_initializer=KERNEL)(u7)
    c7 = ReLU()(c7)
    c7 = BatchNormalization()(c7)
    c7 = Conv2D(256, (3, 3), padding='same', kernel_initializer=KERNEL)(c7)
    c7 = ReLU()(c7)
    c7 = BatchNormalization()(c7)

    u8 = Conv2DTranspose(32, (2, 2), strides=2, padding='same')(c7)
    u8 = tf.concat([u8, c2], axis=3)
    c8 = Conv2D(128, (3, 3), padding='same', kernel_initializer=KERNEL)(u8)
    c8 = ReLU()(c8)
    c8 = BatchNormalization()(c8)
    c8 = Conv2D(128, (3, 3), padding='same', kernel_initializer=KERNEL)(c8)
    c8 = ReLU()(c8)
    c8 = BatchNormalization()(c8)

    u9 = Conv2DTranspose(64, (2, 2), strides=2, padding='same')(c8)
    u9 = tf.concat([c1, u9], axis=3)
    c9 = Conv2D(64, (3, 3), padding='same', kernel_initializer=KERNEL)(u9)
    c9 = ReLU()(c9)
    c9 = BatchNormalization()(c9)
    c9 = Conv2D(64, (3, 3), padding='same', kernel_initializer=KERNEL)(c9)
    c9 = ReLU()(c9)
    c9 = BatchNormalization()(c9)

    # set outputs
    outputs = Conv2D(N_CLASSES, (1, 1), activation='softmax')(c9)
    model = Model(inputs=inputs, outputs=outputs)

    return model
