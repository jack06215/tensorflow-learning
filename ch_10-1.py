import tensorflow as tf
import numpy as np
from tensorflow import keras

print(tf.__version__)
print(np.__version__)
print(keras.__version__)

fashion_mnist = keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, Y_test) = fashion_mnist.load_data()