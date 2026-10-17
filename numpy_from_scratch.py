import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#%matplotlib inline
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing


housing_data = fetch_california_housing()
features = pd.DataFrame(housing_data.data, columns=housing_data.feature_names)
target = pd.DataFrame(housing_data.target, columns=['Target'])
df = features.join(target)
df.head()

scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df[['MedInc','AveRooms','Target']])

X_train, X_test, y_train, y_test = train_test_split(
    df_scaled[:10000, :2], 
    df_scaled[:10000, 2:], 
    test_size=0.33, 
    random_state=42
)

X_train.shape

class SimpleNeuralNetwork:
    
    def __init__(self): 
        
                
        # Initial weights and biases assigned random values ranging from 0 to 1. 
        # We have a total of 9 weights and 4 biases as shown in the figure above.
        
        # Weights
        self.w1, \
        self.w2, \
        self.w3, \
        self.w4, \
        self.w5, \
        self.w6, \
        self.w7, \
        self.w8, \
        self.w9 = np.random.rand(9)

        # Biases
        self.b_n1, \
        self.b_n2, \
        self.b_n3, \
        self.b_y_hat = np.random.rand(4) 

    # Activation functions for the neurons
    # sigmoid is for forward propagation, sigmoid derivative is for back propagation
    
    def sigmoid(self, x): return 1 / (1 + np.e**-x)
    def sigmoid_der(self, x): return self.sigmoid(x) * (1 - self.sigmoid(x))

    
    # Feedforward function produces result of the network prediction for each sample.

    def feedforward(self, x):

        # x[0], x[1] - features
        # n* - neurons in the hidden layer, y_hat - predicted value
        self.n1    = self.sigmoid(x[0]*self.w1 + x[1]*self.w2 + self.b_n1)
        self.n2    = self.sigmoid(x[0]*self.w3 + x[1]*self.w4 + self.b_n2)
        self.n3    = self.sigmoid(x[0]*self.w5 + x[1]*self.w6 + self.b_n3)
        self.y_hat = self.sigmoid(self.n1*self.w7 + self.n2*self.w8 + self.n3*self.w9 + self.b_y_hat)

    

    # Backpropagation updates all the weights and biases of the network.
    # By using Gradient Descent technique, each trainable parameter (weight or bias) is changing a little 
    # bit towards the minimum of MSE.
    # "output layer" => "hidden layer 2" => "hidden layer 1"
    
    def backpropagation(self, x, y):

        # We calculate some values here to use them later
        y_hat_der = (-2 * (y-self.y_hat) * self.sigmoid_der(self.n1*self.w7 + self.n2*self.w8 + self.n3*self.w9 + self.b_y_hat))
        n1_der = self.w7 * self.sigmoid_der(x[0]*self.w1 + x[1]*self.w2 + self.b_n1)
        n2_der = self.w8 * self.sigmoid_der(x[0]*self.w3 + x[1]*self.w4 + self.b_n2)
        n3_der = self.w9 * self.sigmoid_der(x[0]*self.w5 + x[1]*self.w6 + self.b_n3)

        # Biases
        self.b_n1    -= self.lr * y_hat_der * n1_der
        self.b_n2    -= self.lr * y_hat_der * n2_der
        self.b_n3    -= self.lr * y_hat_der * n3_der
        self.b_y_hat -= self.lr * y_hat_der

        # Weights
        self.w7 -= self.lr * y_hat_der * self.n1
        self.w8 -= self.lr * y_hat_der * self.n2
        self.w9 -= self.lr * y_hat_der * self.n3
        
        self.w1 -= self.lr * y_hat_der * n1_der * x[0]
        self.w2 -= self.lr * y_hat_der * n1_der * x[1]
        self.w3 -= self.lr * y_hat_der * n2_der * x[0]
        self.w4 -= self.lr * y_hat_der * n2_der * x[1]
        self.w5 -= self.lr * y_hat_der * n3_der * x[0]
        self.w6 -= self.lr * y_hat_der * n3_der * x[1]
        

    # Training process is the combination of forward and back propagations.
    
    def fit(self, X, y, epoch=10, lr=0.01):

        mse_list = []
        self.lr = lr

        # Loop over epochs. Each epoch train network on all available data.
        # We also store MSE to visualize the training process
        for i in range(epoch):
            mse = mean_squared_error(y, self.predict(X))
            mse_list.append(mse)
            print(f'Epoch: {i+1} / {epoch}, MSE: {round(mse, 4)}')

            # Loop over each training example for current epoch
            for j in range(len(X)):
                self.feedforward(X[j])
                self.backpropagation(X[j], y[j][0])

        return mse_list

    # This function is very similar to feed forward, in fact it uses 'feedforward' function to make predictions.
    # The only difference is that we predict output for all samples.
    
    def predict(self, X):

        result = []

        for x in X:
            self.feedforward(x)
            result.append(self.y_hat)

        return result
    
model = SimpleNeuralNetwork()

history = model.fit(X_train, y_train, epoch=100, lr=0.05)

# Plot the training process
plt.rcParams['figure.dpi'] = 227
plt.rcParams['figure.figsize'] = (16, 6)
plt.plot(range(1, 101), history, label="NN")
plt.title('Neural Network Training Process', fontsize=15)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('MSE', fontsize=12)
plt.legend()
plt.show()
