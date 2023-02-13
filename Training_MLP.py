import pandas as pd

import numpy as np

import matplotlib.pylab as plt

import os
from sklearn.model_selection import train_test_split

import warnings
from Neural_Network import NeuralNetwork
warnings.filterwarnings('ignore')

def one_hot_enc(y):
    predictions = []
    for i in range(len(y)):
        preds = [0,0,0,0]
        max_index = np.argmax(y[i])
        preds[max_index] = 1
        predictions.append(preds)
    return predictions

def accuracy(y_true, y_pred):
    if not (len(y_true) == len(y_pred)):
        print('Size of predicted and true labels not equal.')
        return 0.0

    corr = 0
    for i in range(0,len(y_true)):
        corr += 1 if (y_true[i] == y_pred[i]).all() else 0

    return corr/len(y_true)


train_data_CSV = 'train_data.csv'
train_labels_CSV = 'train_labels.csv'


train_data_path = os.path.join(os.path.dirname(__file__), train_data_CSV)
train_labels_path = os.path.join(os.path.dirname(__file__), train_labels_CSV)




#read the csv file

train_data_np = pd.read_csv(train_data_path, header=None).to_numpy()
train_labels_np = pd.read_csv(train_labels_path, header=None).to_numpy()



#split the data into 80 20
train_x, val_x, train_y, val_y = train_test_split(train_data_np, train_labels_np, test_size = 0.3)

epoch = 100


def train_save_weights(train_data_CSV, train_labels_CSV, epoch, i_s, o_s, h_s, alpha):
    np.random.seed(1)
    
    train_data_path = os.path.join(os.path.dirname(__file__), train_data_CSV)
    train_labels_path = os.path.join(os.path.dirname(__file__), train_labels_CSV)

    #read the csv file

    train_data_np = pd.read_csv(train_data_path, header=None).to_numpy()
    train_labels_np = pd.read_csv(train_labels_path, header=None).to_numpy()

    #split the data into 80 20
    train_x, val_x, train_y, val_y = train_test_split(train_data_np, train_labels_np, test_size = 0.2)
    


            
    NN = NeuralNetwork()

    losses = []
    epochs = []
    for i in range(epoch):
        acc_train, loss = NN.train(train_x, train_y, alpha)
        y_out = NN.predict(val_x)
        acc_val_test = accuracy(val_y, y_out)
        losses.append(loss)
        epochs.append(i)

        if i%5 == 0:
                print("epoch " , i , "train acc ", acc_train , "val_acc", acc_val_test)
    
    
    NN.save_weights()
    plt.plot(epochs, losses)
    plt.savefig(str(epoch)+'_epoch.png')
    
    return acc_train





train_data_CSV = 'train_data.csv'
train_labels_CSV = 'train_labels.csv'
i_s = 784
o_s = 4
h_s = 100
alpha = 0.0001
epoch = [10, 50, 100, 500]
train_acc = []
for x in epoch:
    acc = train_save_weights(train_data_CSV, train_labels_CSV, x, i_s, o_s, h_s, alpha)
    train_acc.append(acc)



    







    







