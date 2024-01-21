import os
import numpy as np  
import cv2 
import tensorflow as tf 
import matplotlib.pyplot as plt

model = tf.keras.models.load_model("handwriten.model")

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test)= mnist.load_data()

loss, accuracy = model.evaluate(x_test, y_test)

print(loss)
print(accuracy)
