import numpy as np
from Neural_Network import NeuralNetwork
import pandas as pd
from acc_calc import accuracy

STUDENT_NAME = ['SRIRAM RANGANATHAN']
STUDENT_ID = ['20986334']

def test_mlp(data_file):
	# Load the test set
	test_data = pd.read_csv(data_file, header=None).to_numpy()
	# START
	Network_A = NeuralNetwork()
	Network_A.load_weights()
	y_pred_out = Network_A.predict(test_data)
	return y_pred_out
    # END


	# Load your network
	# START

	# END


	# Predict test set - one-hot encoded
	# y_pred = ...

	# return y_pred


'''
How we will test your code:

from test_mlp import test_mlp, STUDENT_NAME, STUDENT_ID
from acc_calc import accuracy 

y_pred = test_mlp('./test_data.csv')

test_labels = ...

test_accuracy = accuracy(test_labels, y_pred)*100


y_pred = test_mlp('test_data.csv')

test_labels = pd.read_csv('test_label.csv', header=None).to_numpy()

test_accuracy = accuracy(test_labels, y_pred)*100

print(test_accuracy)'''