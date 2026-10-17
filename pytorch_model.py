import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing

# Load the California housing dataset
housing_data = fetch_california_housing()
features = pd.DataFrame(housing_data.data, columns=housing_data.feature_names)
target = pd.DataFrame(housing_data.target, columns=['Target'])
df = features.join(target)
df.head()

# Scale the features
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df[['MedInc', 'AveRooms', 'Target']])
df_scaled.shape

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    df_scaled[:10000, :2], 
    df_scaled[:10000, 2:], 
    test_size=0.33, 
    random_state=42
)

X_train.shape

# Define a simple neural network using PyTorch
class SimpleNeuralNetwork(nn.Module):
    def __init__(self):
        super(SimpleNeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(2, 3)  # Input layer to hidden layer
        self.fc2 = nn.Linear(3, 1)  # Hidden layer to output layer
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

# Instantiate the model
model = SimpleNeuralNetwork()

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.05)

# Convert numpy arrays to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

# Training loop
history = []  # Initialize an empty list to store MSE values
for epoch in range(100):
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    history.append(loss.item())  # Append the current loss to the history list
    print(f"Epoch: {epoch+1}, Loss: {loss.item()}")

# Evaluate the model on the test set
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_pred_tensor = model(X_test_tensor).detach().numpy()
mse = mean_squared_error(y_test, y_pred_tensor)
print(f"Test MSE: {mse}")

# Plot the training process
# Plot the training process
plt.rcParams['figure.dpi'] = 227
plt.rcParams['figure.figsize'] = (16, 6)
plt.plot(range(1, 101), history, label="PyTorch NN")
plt.title('Neural Network Training Process', fontsize=15)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('MSE', fontsize=12)
plt.legend()
plt.show()

