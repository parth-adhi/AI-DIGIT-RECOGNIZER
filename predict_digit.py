# Test the model with a sample MNIST image
import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import mnist


model = tf.keras.models.load_model('model.h5')


(x_train, y_train), (x_test, y_test) = mnist.load_data()


test_image = x_test[0]  
test_image = np.expand_dims(test_image, axis=-1)  
test_image = np.expand_dims(test_image, axis=0)  
test_image = test_image / 255.0  


prediction = model.predict(test_image)
predicted_digit = np.argmax(prediction)

print(f"Predicted Digit: {predicted_digit}")
