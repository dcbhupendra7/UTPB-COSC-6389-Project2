import math
import random

# Activation functions
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def relu(x):
    return max(0, x)

def tanh(x):
    return (math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x))

# Derivatives of activation functions
def sigmoid_derivative(x):
    return x * (1 - x)

def relu_derivative(x):
    return 1.0 if x > 0 else 0.0

def tanh_derivative(x):
    return 1 - x ** 2

# Cost functions
def mean_squared_error(target, output):
    return 0.5 * (target - output) ** 2

# Parameters for the network
num_inputs = 10
num_hidden_layers = 2  # Reduced number of hidden layers for less clutter
hidden_layer_width = 4  # Reduced number of neurons per hidden layer
num_outputs = 1
neuron_scale = 20
axon_scale = 1

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
        if self.error != 0.0:
            return
        gradient = self.activation_derivative(self.result)
        delta = self.error * gradient
        for in_axon in self.inputs:
            in_axon.input.error += delta * in_axon.weight
            in_axon.weight -= delta * in_axon.input.result * in_axon.output.network.learning_rate
        self.bias -= delta * self.network.learning_rate

    def connect_input(self, in_neuron):
        in_axon = Axon(in_neuron, self, weight=random.uniform(-0.5, 0.5))
        self.inputs.append(in_axon)

    def connect_output(self, out_neuron):
        out_axon = Axon(self, out_neuron, weight=random.uniform(-0.5, 0.5))
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
            # Position the output neurons in the middle of the last hidden layer
            out_neuron.y = (self.hidden_layer_width / 2) * 50 + 100

    def forward_prop(self, inputs):
        # Forward propagate through input, hidden, and output layers
        for in_neuron in self.inputs:
            in_neuron.forward_prop(inputs)
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

    def train(self, data):
        self.forward_prop(data.inputs)
        self.back_prop(data.outputs)

    def test(self, data):
        self.forward_prop(data.inputs)
        return [out.result for out in self.outputs]

class RandData:
    def __init__(self):
        self.inputs = [random.random() for _ in range(num_inputs)]
        self.outputs = [1 if sum(self.inputs) > (num_inputs / 2) else 0]