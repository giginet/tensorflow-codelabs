import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras

mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

def draw():
    plt.imshow(training_images[0])
    print(training_labels[0])
    print(training_images[0])
# draw()

training_images = training_images / 255.0
test_images = test_images / 255.0
