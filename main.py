import random

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt



train_data = pd.read_csv('mnist_train.csv')
test_data = pd.read_csv('mnist_test.csv')

train_data = np.array(train_data)
dev_data = np.array(test_data)
m, n = train_data.shape

t, l = dev_data.shape
print('t : ' + str(t) + ',  l : '+ str(l))


dev_data_T = dev_data.T
Y_dev = dev_data_T[0]
X_dev = dev_data_T[1:n]
X_dev = X_dev / 255.

train_data_T = train_data.T
Y_train = train_data_T[0]
X_train = train_data_T[1:n]
X_train = X_train / 255.
_,m_train = X_train.shape


def init_params():
    W1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10, 1) - 0.5
    W2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    return W1, b1, W2, b2


def ReLU(Z):
    return np.maximum(Z, 0)


def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A


def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2


def ReLU_deriv(Z):
    return Z > 0


def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y


def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2)

    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1)

    return dW1, db1, dW2, db2


def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    return W1, b1, W2, b2



def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size


def gradient_descent(X, Y, alpha, iterations):
    W1, b1, W2, b2 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i % 100 == 0:
            print("Iteration: ", i)
            predictions = get_predictions(A2)
            print("Accuracy = " + str(get_accuracy(predictions, Y)))
#            alpha = alpha - (get_accuracy(predictions, Y) / 500)
#            print('alpha = ' + str(alpha))
    return W1, b1, W2, b2


W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 0.15, 500)

#print('W1, b1, W2, b2 ' + str(W1), str(b1), str(W2), str(b2))





def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    return predictions


def test_prediction(index, W1, b1, W2, b2):
    print("test prediction")
    current_image = X_dev[:, index, None]
    prediction = make_predictions(X_dev[:, index, None], W1, b1, W2, b2)
    label = Y_dev[index]
    print("Prediction: ", prediction)
    print("Label: ", label)

    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()

def verif_pred_test(W1, b1, W2, b2, index):
    k = 0
    o = 0
    for i in range(index):
        _, _, _, A2 = forward_prop(W1, b1, W2, b2, X_dev[:, i, None])
        test = get_predictions(A2)
        if test == Y_dev[i]:
            k = k + 1
        elif o < 3:
            test_prediction(i, W1, b1, W2, b2)
            o = o + 1
    print("Acuratete teste dev = " + str(k/index))

verif_pred_test(W1, b1, W2, b2, 1000)

j = random.randint(0, t)
#test_prediction(j, W1, b1, W2, b2)
j = random.randint(0, t)
#test_prediction(j, W1, b1, W2, b2)
j = random.randint(0, t)
#test_prediction(j, W1, b1, W2, b2)
j = random.randint(0, t)
#test_prediction(j, W1, b1, W2, b2)


