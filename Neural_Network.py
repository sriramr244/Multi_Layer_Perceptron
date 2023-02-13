from acc_calc import accuracy
import numpy as np

def one_hot_enc(y):
    predictions = []
    for i in range(len(y)):
        preds = [0,0,0,0]
        max_index = np.argmax(y[i])
        preds[max_index] = 1
        predictions.append(preds)
    return predictions

class NeuralNetwork(object):
    def __init__(self, i_s = 784, o_s = 4, h_s = 128):
        self.input_size = i_s
        self.output_size = o_s
        self.hidden_size = h_s

        self.w1 = np.random.randn(self.input_size, self.hidden_size)
        self.w2 = np.random.randn(self.hidden_size, self.output_size)
        self.b1 = np.random.randn(self.hidden_size)
        self.b2 = np.random.randn(self.output_size)

    def sigmoid(self, s, deriv = False):
        if deriv == True:
            return self.sigmoid(s)*(1-self.sigmoid(s))
        return 1/(1+np.exp(-s))

    def feed_forward(self, X):
        self.z1 = np.dot(X, self.w1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.w2) +self.b2
        self.a2 = self.Pred_output = self.softmax(self.z2)
        return self.Pred_output
    
    
    
    def softmax(self,x):
        # try:
        #     self.e = np.exp(x)
        # except:
        #     print(x)
        return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)
    #, axis=1, keepdims=True

    def back_prop(self, X, y, Output):
        self.output_err = Output - y
        self.del_b2 = self.output_err
        self.del_w2 = np.dot(self.a1.T,self.output_err)
        self.lay_err = np.dot(self.output_err, self.w2.T)
        self.del_b1 = self.lay_err * self.sigmoid(self.z1, deriv = True)
        self.del_w1 = np.dot(X.T, self.sigmoid(self.z1, deriv = True) * self.lay_err)
        return

    def update_weights(self, alpha):
        self.w1 = self.w1 - alpha * self.del_w1
        self.w2 = self.w2 - alpha * self.del_w2
        self.b1 = self.b1 - alpha * self.del_b1.sum(axis=0)
        self.b2 = self.b2 - alpha * self.del_b2.sum(axis=0)
        #print(self.w1, self.w2, self.b1, self.b2)
        return
    
    def one_hot_enc(y):
        predictions = []
        for i in range(len(y)):
            preds = [0,0,0,0]
            max_index = np.argmax(y[i])
            preds[max_index] = 1
            predictions.append(preds)
        return predictions


    def train(self,X,y, alpha):
        output = self.feed_forward(X)

        self.back_prop(X, y, output)
        self.update_weights(alpha)
        p_output = self.feed_forward(X)
        y_train_pred = one_hot_enc(p_output)
        cost = -np.mean(y * np.log(output+ 1e-8))
        acc_train = accuracy(y, y_train_pred)
        return acc_train , cost
    
    def predict(self, X, LD_wts = False):
        output = self.feed_forward(X)
        y_out = one_hot_enc(output)
        return y_out

    def save_weights(self):
        # save the model to disk
        
        np.save('w1.npy' , self.w1)
        np.save('w2.npy' , self.w2)
        np.save('b1.npy' , self.b1)
        np.save('b2.npy' , self.b2)
    
    def load_weights(self):
        self.w1 = np.load('w1.npy')
        self.w2 = np.load('w2.npy')
        self.b1 = np.load('b1.npy')
        self.b2 = np.load('b2.npy')