import matplotlib.pyplot as plt
import pandas as pd

import os
train_data_CSV = 'train_data.csv'
train_labels_CSV = 'train_labels.csv'

train_data_path = os.path.join(os.path.dirname(__file__), train_data_CSV)
train_labels_path = os.path.join(os.path.dirname(__file__), train_labels_CSV)

#read the csv file

train_data_np = pd.read_csv(train_data_path, header=None).to_numpy()
train_labels_np = pd.read_csv(train_labels_path, header=None).to_numpy()

