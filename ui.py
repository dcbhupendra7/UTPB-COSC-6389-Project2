import tkinter as tk
import customtkinter as ctk
from tkinter import Spinbox, messagebox
import threading
import pandas as pd
import numpy as np
from neural_network import Network, load_and_preprocess_data, calculate_r2, calculate_mae

def start_training():
    global accuracy_label, learning_rate_spinbox, input_neurons_spinbox, hidden_layers_spinbox, layer_width_spinbox, activation_var, iteration_label, canvas, dataset_description_label, training_status_label, network_accuracy_label

    activation = activation_var.get()
    if activation not in ['sigmoid', 'relu', 'tanh']:
        messagebox.showerror("Error", "Invalid activation function.")
        return

    # Validate all input fields
    is_valid = True
    try:
        learning_rate = float(learning_rate_spinbox.get())
    except ValueError:
        learning_rate_spinbox.config(fg="red")
        is_valid = False

    try:
        num_inputs = int(input_neurons_spinbox.get())
    except ValueError:
        input_neurons_spinbox.config(fg="red")
        is_valid = False

    try:
        num_hidden_layers = int(hidden_layers_spinbox.get())
    except ValueError:
        hidden_layers_spinbox.config(fg="red")
        is_valid = False

    try:
        hidden_layer_width = int(layer_width_spinbox.get())
    except ValueError:
        layer_width_spinbox.config(fg="red")
        is_valid = False

    if not is_valid:
        return

    learning_rate_spinbox.config(fg="black")
    input_neurons_spinbox.config(fg="black")
    hidden_layers_spinbox.config(fg="black")
    layer_width_spinbox.config(fg="black")

    # Load and preprocess the dataset
    X_train, X_test, y_train, y_test = load_and_preprocess_data("student_sleep_patterns.csv")

    dataset_description_label.config(text=f"Dataset Overview:\n"
                                      f" • Features: {X_train.shape[1]} (e.g., sleep duration, study hours)\n"
                                      f" • Training Samples: {len(X_train)}\n"
                                      f" • Test Samples: {len(X_test)}\n"
                                      f" • Purpose: Understanding the relationship between sleep duration and academic performance (proxied by study hours)\n"
                                      f" • R² Score: Measures how well the predictions approximate the actual values. A value closer to 1 indicates a better fit.\n"
                                      f" • Mean Absolute Error (MAE): Average error magnitude in predictions. Lower values are better.",
                                   fg="black")

    if num_inputs > X_train.shape[1]:
        messagebox.showerror("Error", f"Number of input neurons cannot exceed {X_train.shape[1]}.")
        return

    # Re-initialize the network on each training to ensure different starting weights
    network = Network(num_inputs=num_inputs, num_hidden_layers=num_hidden_layers,
                      hidden_layer_width=hidden_layer_width, learning_rate=learning_rate)
        # Reinitialize weights by creating a new instance of the network for each training cycle

    # Update status to indicate training has started
    training_status_label.config(text="Training in progress...", fg="black")

    def run_training():
        # Shuffle the dataset on every training cycle
        nonlocal X_train, y_train
        shuffled_indices = np.random.permutation(len(X_train))
        X_train, y_train = X_train[shuffled_indices], y_train[shuffled_indices]
        for epoch in range(100):
            network.train(X_train, y_train, epochs=1)
            predictions = network.predict(X_test)
            
            # Calculate Mean Squared Error
            mse = ((predictions - y_test) ** 2).mean()

            # Calculate R² score for evaluation
            r2 = calculate_r2(predictions, y_test)

            # Calculate Mean Absolute Error
            mae = calculate_mae(predictions, y_test)

            # Update UI elements
            accuracy_label.config(text=f"Final Mean Squared Error: {mse:.4f}\nR² Score: {r2:.4f}", fg="black")
            network_accuracy_label.config(text=f"Mean Absolute Error (MAE): {mae:.4f}", fg="black")
            iteration_label.config(text=f"Iteration: {epoch + 1} / 100", fg="black")
            training_status_label.config(text="Training Complete!" if epoch == 99 else f"Training... {epoch + 1}/100", fg="black")
            
            # Update network visualization
            canvas.after(0, lambda: draw_network(network))

    training_thread = threading.Thread(target=run_training)
    training_thread.start()

def train():
    global accuracy_label, learning_rate_spinbox, input_neurons_spinbox, hidden_layers_spinbox, layer_width_spinbox, activation_var, iteration_label, canvas, dataset_description_label, training_status_label, network_accuracy_label

    root = tk.Tk()
    root.title("Neural Network Trainer")
    root.configure(bg="white")
    
    # Make window fullscreen
    root.state('zoomed')  # For Windows
    # For Linux/Mac, uncomment the following line instead:
    # root.attributes('-zoomed', True)

    # Main container with padding
    main_container = tk.Frame(root, bg="white", padx=20, pady=20)
    main_container.pack(expand=True, fill="both")

    # Top row container
    top_row = tk.Frame(main_container, bg="white")
    top_row.pack(fill="x", pady=(0, 20))
    top_row.grid_columnconfigure(0, weight=1)
    top_row.grid_columnconfigure(1, weight=1)

    # Dataset Overview Card with larger font
    dataset_frame = tk.LabelFrame(top_row, text="Dataset and Training Overview", 
                                font=("Arial", 16, "bold"), bg="#E8F5E9", 
                                fg="black", padx=20, pady=20)
    dataset_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10))

    dataset_description_label = tk.Label(dataset_frame, 
                                       text="Dataset Overview: Not loaded yet",
                                       bg="#E8F5E9", justify="left",
                                       font=("Arial", 14), fg="black")
    dataset_description_label.pack(anchor="w")

    training_goal_label = tk.Label(dataset_frame,
                                 text="Training Goal: Minimize Mean Squared Error (MSE)",
                                 bg="#E8F5E9", justify="left",
                                 font=("Arial", 14), fg="black")
    training_goal_label.pack(anchor="w", pady=(10, 0))

    # Training Configuration Card with larger font
    config_frame = tk.LabelFrame(top_row, text="Training Configuration",
                               font=("Arial", 16, "bold"), bg="#E3F2FD",
                               fg="black", padx=20, pady=20)
    config_frame.grid(row=0, column=1, sticky="nsew", padx=(10, 0))

    # Configuration inputs with larger font
    input_configs = [
        ("Activation Function:", "sigmoid", "menu"),
        ("Learning Rate:", "0.1", "spinbox"),
        ("Number of Input Neurons:", "4", "spinbox"),
        ("Number of Hidden Layers:", "2", "spinbox"),
        ("Neurons per Hidden Layer:", "4", "spinbox")
    ]

    for i, (label_text, default_value, input_type) in enumerate(input_configs):
        label = tk.Label(config_frame, text=label_text, bg="#E3F2FD",
                        font=("Arial", 14), fg="black")
        label.grid(row=i, column=0, pady=5, padx=5, sticky="e")

        if input_type == "menu":
            activation_var = tk.StringVar(value=default_value)
            menu = tk.OptionMenu(config_frame, activation_var, "sigmoid", "relu", "tanh")
            menu.config(width=15, bg="white", fg="black", font=("Arial", 12))
            menu["menu"].config(bg="white", fg="black", font=("Arial", 12))
            menu.grid(row=i, column=1, pady=5, padx=5, sticky="w")
        else:
            spinbox = Spinbox(config_frame, from_=0.01 if label_text.startswith("Learning") else 1,
                            to=1.0 if label_text.startswith("Learning") else 100,
                            increment=0.01 if label_text.startswith("Learning") else 1,
                            width=10, fg="black", bg="white",
                            font=("Arial", 12),
                            highlightbackground="white",
                            highlightcolor="white")
            spinbox.delete(0, "end")
            spinbox.insert(0, default_value)
            spinbox.grid(row=i, column=1, pady=5, padx=5, sticky="w")
            
            if label_text.startswith("Learning"):
                learning_rate_spinbox = spinbox
            elif "Input" in label_text:
                input_neurons_spinbox = spinbox
            elif "Hidden Layers" in label_text:
                hidden_layers_spinbox = spinbox
            else:
                layer_width_spinbox = spinbox

    # Network Visualization with increased height
    canvas = tk.Canvas(main_container, bg="white", height=400)
    canvas.pack(fill="both", pady=(0,40))

    # Bottom row container
    bottom_row = tk.Frame(main_container, bg="white")
    bottom_row.pack(fill="x")
    bottom_row.grid_columnconfigure(0, weight=1)
    bottom_row.grid_columnconfigure(1, weight=1)

    # Start Training Button with larger font
    start_button = ctk.CTkButton(bottom_row, text="Start Training",
                                command=start_training,
                                fg_color="#FF6F61",
                                text_color="white",
                                hover_color="#FF856C",
                                height=60, width=200,
                                font=("Arial", 18, "bold"))
    start_button.grid(row=0, column=0, pady=10, sticky="e", padx=10)

    # Training Results Card with larger font
    results_frame = tk.LabelFrame(bottom_row, text="Training Results",
                                font=("Arial", 16, "bold"), bg="#FFF4E6",
                                fg="black", padx=20, pady=20)
    results_frame.grid(row=0, column=1, sticky="w", padx=10)

    training_status_label = tk.Label(results_frame, text="Waiting to start training...",
                                   font=("Arial", 14), bg="#FFF4E6", fg="black")
    training_status_label.pack()

    accuracy_label = tk.Label(results_frame, text="Final Mean Squared Error: 0.0000\nR² Score: 0.0000",
                            font=("Arial", 18, "bold"), bg="#FFF4E6", fg="black")
    accuracy_label.pack(pady=5)

    network_accuracy_label = tk.Label(results_frame, text="Mean Absolute Error (MAE): 0.0000",
                                      font=("Arial", 18, "bold"), bg="#FFF4E6", fg="black")
    network_accuracy_label.pack(pady=5)

    iteration_label = tk.Label(results_frame, text="Iteration: 0 / 100",
                             font=("Arial", 14), bg="#FFF4E6", fg="black")
    iteration_label.pack()

    root.update_idletasks()  # Ensure all widgets are rendered properly
    root.mainloop()

def draw_network(network):
    canvas.delete("all")
    
    # Increased dimensions
    layer_x = 100
    layer_gap = 250  # Increased gap between layers
    neuron_gap = 80  # Increased gap between neurons
    neuron_radius = 25  # Increased neuron size
    
    # Calculate total height needed for the network
    max_neurons_in_layer = max(len(network.inputs), 
                             max(len(layer) for layer in network.hidden_layers),
                             len(network.outputs))
    
    total_height = (max_neurons_in_layer - 1) * neuron_gap
    starting_y = 50  # Start from top with some padding

    # Helper function to draw a neuron
    def draw_neuron(x, y, value):
        canvas.create_oval(x - neuron_radius, y - neuron_radius,
                         x + neuron_radius, y + neuron_radius,
                         fill="white", outline="black", width=2)

    # Draw input layer
    input_neurons = {}
    for i, neuron in enumerate(network.inputs):
        y = starting_y + i * neuron_gap
        draw_neuron(layer_x, y, neuron.result)
        input_neurons[neuron] = (layer_x, y)

    # Draw hidden layers
    hidden_neurons = {}
    current_x = layer_x + layer_gap
    for layer in network.hidden_layers:
        layer_height = (len(layer) - 1) * neuron_gap
        layer_start_y = starting_y + (total_height - layer_height) / 2
        
        for i, neuron in enumerate(layer):
            y = layer_start_y + i * neuron_gap
            draw_neuron(current_x, y, neuron.result)
            hidden_neurons[neuron] = (current_x, y)
        current_x += layer_gap

    # Draw output layer (centered vertically relative to the last hidden layer)
    output_neurons = {}
    output_height = (len(network.outputs) - 1) * neuron_gap
    output_start_y = starting_y + (total_height - output_height) / 2
    
    for i, neuron in enumerate(network.outputs):
        y = output_start_y + i * neuron_gap
        draw_neuron(current_x, y, neuron.result)
        output_neurons[neuron] = (current_x, y)

    # Draw connections
    def draw_connections(source_neurons, next_layer):
        for source in source_neurons:
            sx, sy = source_neurons[source]
            for axon in source.outputs:
                if axon.output in next_layer:
                    tx, ty = next_layer[axon.output]
                    canvas.create_line(sx + neuron_radius, sy,
                                    tx - neuron_radius, ty,
                                    fill="gray", width=1)
                    # Draw weight in middle of line
                    mx, my = (sx + tx) / 2, (sy + ty) / 2
                    canvas.create_text(mx, my, text=f"{axon.weight:.2f}",
                                    fill="red", font=("Arial", 10))

    # Draw all connections
    for i in range(len(network.hidden_layers)):
        if i == 0:
            draw_connections(input_neurons, hidden_neurons)
        draw_connections(hidden_neurons, hidden_neurons)
    draw_connections(hidden_neurons, output_neurons)

if __name__ == '__main__':
    train()