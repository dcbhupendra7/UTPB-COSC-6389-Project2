import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import random

# Activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def tanh(x):
    return np.tanh(x)

# Derivatives of activation functions
def sigmoid_derivative(x):
    return x * (1 - x)

def relu_derivative(x):
    return np.where(x > 0, 1.0, 0.0)

def tanh_derivative(x):
    return 1 - x ** 2

# Xavier initialization function
def xavier_init(num_inputs, hidden_layer_width):
    return random.uniform(-1.0, 1.0) * np.sqrt(2 / (num_inputs + hidden_layer_width))

# Parameters for the network
num_inputs = 10
num_hidden_layers = 2
hidden_layer_width = 4
num_outputs = 1
training_data_size = 1000

# Neuron class with selectable activation function
class Neuron:
    def __init__(self, x, y, activation='sigmoid', input_idx=-1, bias=0.0, network=None):
        self.network = network
        self.x = x
        self.y = y
        self.activation = activation
        self.inputs = []
        self.outputs = []
        self.index = input_idx
        self.bias = bias
        self.result = 0.0
        self.error = 0.0

    def activate(self, total):
        if self.activation == 'sigmoid':
            return sigmoid(total)
        elif self.activation == 'relu':
            return relu(total)
        elif self.activation == 'tanh':
            return tanh(total)
        else:
            raise ValueError("Unknown activation function")

    def activation_derivative(self, value):
        if self.activation == 'sigmoid':
            return sigmoid_derivative(value)
        elif self.activation == 'relu':
            return relu_derivative(value)
        elif self.activation == 'tanh':
            return tanh_derivative(value)
        else:
            raise ValueError("Unknown activation function")

    def forward_prop(self, inputs):
        if self.index >= 0:
            self.result = inputs[self.index]
        else:
            total = sum(in_axon.weight * in_axon.input.result for in_axon in self.inputs) + self.bias
            self.result = self.activate(total)

    def back_prop(self):
        # Ensure delta is initialized properly
        if isinstance(self.error, (int, float, np.float64)):
            if self.error != 0.0:
                gradient = self.activation_derivative(self.result)
                delta = self.error * gradient

                for in_axon in self.inputs:
                    in_axon.input.error += delta * in_axon.weight
                    in_axon.weight -= delta * in_axon.input.result * self.network.learning_rate
                self.bias -= delta * self.network.learning_rate

    def connect_input(self, in_neuron):
        in_axon = Axon(in_neuron, self, weight=xavier_init(num_inputs, hidden_layer_width))
        self.inputs.append(in_axon)

    def connect_output(self, out_neuron):
        out_axon = Axon(self, out_neuron, weight=xavier_init(num_inputs, hidden_layer_width))
        self.outputs.append(out_axon)

class Axon:
    def __init__(self, in_n, out_n, weight=0.0):
        self.input = in_n
        self.output = out_n
        self.weight = weight

class Network:
    def __init__(self, activation='sigmoid', num_hidden_layers=2, hidden_layer_width=4, learning_rate=0.1, num_inputs=10):
        self.learning_rate = learning_rate
        self.num_hidden_layers = num_hidden_layers
        self.hidden_layer_width = hidden_layer_width
        self.num_inputs = num_inputs
        self.inputs = [Neuron(0, 0, input_idx=i, network=self) for i in range(self.num_inputs)]
        self.hidden_layers = [[Neuron(0, 0, activation=activation, network=self) for _ in range(self.hidden_layer_width)] for _ in range(self.num_hidden_layers)]
        self.outputs = [Neuron(0, 0, activation='sigmoid', network=self) for _ in range(num_outputs)]
        self.connect_layers()

    def connect_layers(self):
        for layer_idx, layer in enumerate(self.hidden_layers):
            for neuron in layer:
                if layer_idx == 0:
                    for in_neuron in self.inputs:
                        neuron.connect_input(in_neuron)
                        in_neuron.connect_output(neuron)
                else:
                    for prev_neuron in self.hidden_layers[layer_idx - 1]:
                        neuron.connect_input(prev_neuron)
                        prev_neuron.connect_output(neuron)
        for out_neuron in self.outputs:
            for last_hidden_neuron in self.hidden_layers[-1]:
                out_neuron.connect_input(last_hidden_neuron)
                last_hidden_neuron.connect_output(out_neuron)

    def forward_prop(self, inputs):
        # Propagate forward for each sample in the batch
        for sample in inputs:
            for in_neuron, value in zip(self.inputs, sample):
                in_neuron.result = value
            for layer in self.hidden_layers:
                for neuron in layer:
                    neuron.forward_prop(inputs)
            for out_neuron in self.outputs:
                out_neuron.forward_prop(inputs)

    def back_prop(self, target_outputs):
        for out_neuron, target in zip(self.outputs, target_outputs):
            out_neuron.error = target - out_neuron.result
        for layer in reversed(self.hidden_layers):
            for neuron in layer:
                neuron.back_prop()

    def train(self, X_train, y_train, epochs=1):
        for _ in range(epochs):
            for inputs, target in zip(X_train, y_train):
                self.forward_prop([inputs])
                self.back_prop([target])

    def predict(self, X):
        predictions = []
        for inputs in X:
            self.forward_prop([inputs])
            predictions.append(self.outputs[0].result)
        return np.array(predictions)

# Load and preprocess data
def load_and_preprocess_data(file_path):
    # Load dataset
    df = pd.read_csv(file_path)

    # Drop Student_ID as it's not useful for training
    df.drop(columns=['Student_ID'], inplace=True)

    # Encode categorical variables
    label_encoders = {}
    for column in ['Gender', 'University_Year']:
        label_encoders[column] = LabelEncoder()
        df[column] = label_encoders[column].fit_transform(df[column])

    # Normalize numerical features
    scaler = MinMaxScaler()
    numerical_features = ['Age', 'Sleep_Duration', 'Study_Hours', 'Screen_Time', 'Caffeine_Intake', 'Physical_Activity', 
                          'Weekday_Sleep_Start', 'Weekend_Sleep_Start', 'Weekday_Sleep_End', 'Weekend_Sleep_End']
    df[numerical_features] = scaler.fit_transform(df[numerical_features])

    # Separate inputs and outputs
    X = df.drop(columns=['Sleep_Quality']).values
    y = df['Sleep_Quality'].values

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

# Calculate accuracy
def calculate_accuracy(predictions, targets):
    predictions = np.round(predictions)  # Assuming it's a regression output that should be rounded
    correct = np.sum(predictions == targets)
    return (correct / len(targets)) * 100
