import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense, Flatten
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np

from ann import ANN
from cnn import CNN

# (X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()
# X_train = keras.utils.normalize(X_train, axis=1)
# X_test = keras.utils.normalize(X_test, axis=1)


# model = Sequential()
# model.add(Flatten(input_shape=(28, 28)))
# print(X_train[0][0].shape)

# model.add(Dense(128, activation="relu"))
# model.add(Dense(10, activation="softmax"))


# model.compile(
#     optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
# )

# history = model.fit(
#     x=X_train, y=y_train, epochs=10, shuffle="True", validation_split=0.2
# )

# result = model.evaluate(X_test, y_test, batch_size=10)

# print("test Loss : ", result[0], " , Test Accuracy : ", result[1])


# accuracy = accuracy_score(y_test, predictions)
# print(accuracy)

(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
print("training set : ", X_train.shape, "  Dimention od each image:", X_train[0].shape)

# Normalize because rgb values are ranging from 0 to 255

X_train = tf.keras.utils.normalize(X_train, axis=1)
X_test = tf.keras.utils.normalize(X_test, axis=1)
plt.imshow(X_train[0])

print("training set : ", X_train.shape, "  Dimention od each image:", X_train[0].shape)

IMG_SIZE = 28

X_trainr = np.array(X_train).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
X_testr = np.array(X_test).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

model = ANN(X_train, y_train, X_test, y_test, 5)

y_predict = model.predict(X_test)
predictions = y_predict.argmax(axis=1)
print("Predicted : ", predictions, " Actual:", y_test)

CNN(X_trainr, y_train, X_testr, y_test, 5)
