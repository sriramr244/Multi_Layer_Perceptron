
import numpy as np




def sigmoid(x):
	return 1/(1+np.exp(-x))

def softmax(x):
#     https://stackoverflow.com/questions/47372685/softmax-function-in-neural-network-python
    e = np.exp(x)
    return e / np.sum(e, axis=1, keepdims=True)

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


def forward_pop(x_inp, w1, w2, b1, b2):
	z1 = np.dot(x_inp, w1) + b1
	a1 = sigmoid(z1)
	z2 = np.dot(a1, w2) + b2
	a2 = softmax(z2)
	return a2,z2,a1,z1


def backprop(output, z1, a1, w2,b1,b2, train_y, train_x):
    del_loss = output - train_y
    del_b2 = del_loss
    del_w2 = np.dot(a1.T, del_loss)
    lay_err = np.dot(del_loss, w2.T)
    del_w1 = np.dot(train_x.T, sigmoid_derivative(z1) * lay_err)
    del_b1 = lay_err * sigmoid_derivative(z1)
    return del_b2, del_w2, del_w1, del_b1