import tensorflow as tf
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

# Define a simple neural network using TensorFlow
class SimpleNeuralNetwork(tf.keras.Model):
    def __init__(self):
        super(SimpleNeuralNetwork, self).__init__()
        self.fc1 = tf.keras.layers.Dense(3, activation='sigmoid')  # Hidden layer
        self.fc2 = tf.keras.layers.Dense(1, activation='sigmoid')  # Output layer

    def call(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x

# Instantiate the model
model = SimpleNeuralNetwork()

# Define loss function and optimizer
criterion = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.SGD(learning_rate=0.05)

# Convert numpy arrays to TensorFlow tensors
X_train_tensor = tf.convert_to_tensor(X_train, dtype=tf.float32)
y_train_tensor = tf.convert_to_tensor(y_train, dtype=tf.float32)

# Training loop
history = []  # Initialize an empty list to store MSE values
for epoch in range(100):
    with tf.GradientTape() as tape:
        outputs = model(X_train_tensor)
        loss = criterion(y_train_tensor, outputs)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    history.append(loss.numpy())
    print(f"Epoch: {epoch+1}, Loss: {loss.numpy()}")

# Evaluate the model on the test set
X_test_tensor = tf.convert_to_tensor(X_test, dtype=tf.float32)
y_pred_tensor = model(X_test_tensor).numpy()
mse = mean_squared_error(y_test, y_pred_tensor)
print(f"Test MSE: {mse}")

# Plot the training process
plt.rcParams['figure.dpi'] = 227
plt.rcParams['figure.figsize'] = (16, 6)
plt.plot(range(1, 101), history, label="TensorFlow NN")
plt.title('Neural Network Training Process', fontsize=15)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('MSE', fontsize=12)
plt.legend()
plt.show()
