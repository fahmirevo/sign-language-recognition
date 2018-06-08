from keras.models import load_model
import numpy as np

X = np.load("dataset/X_data1.npy")
Y = np.load("dataset/Y_data1.npy")

X = X.reshape(X.shape + (1,))

model = load_model("cnn.model")

score = model.evaluate(X, Y)

print(score[0], score[1])

# print(np.argmax(model.predict(X[:200]), axis=1))
# print(np.argmax(model.predict(X), axis=1) == np.argmax(Y, axis=1))
# print(model.predict(X[:50]))
