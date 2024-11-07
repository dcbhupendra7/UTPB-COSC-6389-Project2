import customtkinter as ctk
from tkinter import messagebox, Spinbox
from neural_network import Network, RandData, neuron_scale, axon_scale, training_data_size
import threading

def train():
    def start_training():
        activation = activation_var.get()
        if activation not in ['sigmoid', 'relu', 'tanh']:
            messagebox.showerror("Error", "Invalid activation function.")
            return

        global learning_rate, num_hidden_layers, hidden_layer_width, num_inputs
        is_valid = True
        try:
            learning_rate = float(learning_rate_spinbox.get())
        except ValueError:
            learning_rate_spinbox.configure(foreground="red")
            is_valid = False

        try:
            num_inputs = int(input_neurons_spinbox.get())
        except ValueError:
            input_neurons_spinbox.configure(foreground="red")
            is_valid = False

        try:
            num_hidden_layers = int(hidden_layers_spinbox.get())
        except ValueError:
            hidden_layers_spinbox.configure(foreground="red")
            is_valid = False

        try:
            hidden_layer_width = int(layer_width_spinbox.get())
        except ValueError:
            layer_width_spinbox.configure(foreground="red")
            is_valid = False

        # If any input is invalid, do not proceed
        if not is_valid:
            return

        # If all inputs are valid, reset foreground color
        learning_rate_spinbox.configure(foreground="black")
        input_neurons_spinbox.configure(foreground="black")
        hidden_layers_spinbox.configure(foreground="black")
        layer_width_spinbox.configure(foreground="black")

        network = Network(activation=activation, num_hidden_layers=num_hidden_layers,
                          hidden_layer_width=hidden_layer_width, learning_rate=learning_rate, num_inputs=num_inputs)
        canvas.delete("all")  
        # Clear canvas before drawing
        update_canvas_size(network)
        draw_network(network)

        training_complete_label.grid_forget()  
        # Hide the training complete label if already visible

        def run_training():
            for i in range(training_data_size):
                data = RandData()
                network.train(data)
                if i % 100 == 0:  
                    # Update network visualization every 100 iterations to reduce lag
                    canvas.after(0, lambda: draw_network(network))

            canvas.after(0, lambda: show_training_complete())

        # Start training in a separate thread
        training_thread = threading.Thread(target=run_training)
        training_thread.start()

    def update_canvas_size(network):
        max_width = 50 + (len(network.hidden_layers) + 2) * 150  
        # Input, hidden layers, output
        max_height = 100 + max(len(network.inputs), len(network.outputs), hidden_layer_width) * 50
        canvas.config(scrollregion=(0, 0, max_width, max_height))
        canvas.configure(width=min(1200, max_width), height=min(800, max_height))

    def draw_network(network):
        layer_x = 50
        layer_gap = 150
        neuron_gap = 50

        # Draw input layer
        for i, neuron in enumerate(network.inputs):
            neuron.x = layer_x
            neuron.y = 100 + i * neuron_gap
            draw_neuron(neuron)

        # Draw hidden layers
        for layer_idx, layer in enumerate(network.hidden_layers):
            layer_x += layer_gap
            for i, neuron in enumerate(layer):
                neuron.x = layer_x
                neuron.y = 100 + i * neuron_gap
                draw_neuron(neuron)
                for in_axon in neuron.inputs:
                    draw_axon(in_axon)

        # Draw output layer
        layer_x += layer_gap
        for i, neuron in enumerate(network.outputs):
            neuron.x = layer_x
            # Position output neuron in the middle of the hidden layer
            neuron.y = (network.hidden_layer_width / 2) * neuron_gap + 100
            draw_neuron(neuron)
            for in_axon in neuron.inputs:
                draw_axon(in_axon)

    def draw_neuron(neuron):
        intensity = int(neuron.result * 255)
        color = f'#{intensity:02x}{intensity:02x}ff'
        canvas.create_oval(neuron.x - neuron_scale, neuron.y - neuron_scale,
                           neuron.x + neuron_scale, neuron.y + neuron_scale,
                           fill="white", outline="black", width=2)
        canvas.create_text(neuron.x, neuron.y, text=f"{neuron.result:.2f}", font=("Arial", 10, "bold"), fill="black")

    def draw_axon(axon):
        line_thickness = max(1, int(abs(axon.weight) * 2))  
        # Reduced line thickness to make visualization clearer 
        canvas.create_line(axon.input.x + neuron_scale * 0.6, axon.input.y, axon.output.x - neuron_scale * 0.6, axon.output.y, width=line_thickness, fill="#444")
        mid_x = (axon.input.x + axon.output.x) / 2
        mid_y = (axon.input.y + axon.output.y) / 2
        canvas.create_text(mid_x, mid_y, text=f"{axon.weight:.2f}", font=("Arial", 8), fill="red")  
        # Decreased font size for better visibility

    def show_training_complete():
        training_complete_label.grid(row=6, column=0, columnspan=2, padx=10, pady=10)  # Show the training complete label

    ctk.set_appearance_mode("light")
    ctk.set_default_color_theme("blue")

    root = ctk.CTk()
    root.title("Neural Network Trainer")

    ctk.CTkLabel(root, text="Activation Function:").grid(row=0, column=0, padx=10, pady=5, sticky="e")
    activation_var = ctk.StringVar(value="sigmoid")
    activation_menu = ctk.CTkComboBox(root, variable=activation_var, values=["sigmoid", "relu", "tanh"])
    activation_menu.grid(row=0, column=1, padx=10, pady=5)

    ctk.CTkLabel(root, text="Learning Rate:").grid(row=1, column=0, padx=10, pady=5, sticky="e")
    learning_rate_spinbox = Spinbox(root, from_=0.01, to=1.0, increment=0.01, bg="#f0f0f0", fg="black", highlightbackground="gray", highlightthickness=1)
    learning_rate_spinbox.delete(0, "end")
    learning_rate_spinbox.insert(0, "0.1")
    learning_rate_spinbox.grid(row=1, column=1, padx=10, pady=5)

    ctk.CTkLabel(root, text="Number of Input Neurons:").grid(row=2, column=0, padx=10, pady=5, sticky="e")
    input_neurons_spinbox = Spinbox(root, from_=1, to=100, increment=1, bg="#f0f0f0", fg="black", highlightbackground="gray", highlightthickness=1)
    input_neurons_spinbox.delete(0, "end")
    input_neurons_spinbox.insert(0, "8")
    input_neurons_spinbox.grid(row=2, column=1, padx=10, pady=5)

    ctk.CTkLabel(root, text="Number of Hidden Layers:").grid(row=3, column=0, padx=10, pady=5, sticky="e")
    hidden_layers_spinbox = Spinbox(root, from_=1, to=10, increment=1, bg="#f0f0f0", fg="black", highlightbackground="gray", highlightthickness=1)
    hidden_layers_spinbox.delete(0, "end")
    hidden_layers_spinbox.insert(0, "1")
    hidden_layers_spinbox.grid(row=3, column=1, padx=10, pady=5)

    ctk.CTkLabel(root, text="Neurons per Hidden Layer:").grid(row=4, column=0, padx=10, pady=5, sticky="e")
    layer_width_spinbox = Spinbox(root, from_=1, to=50, increment=1, bg="#f0f0f0", fg="black", highlightbackground="gray", highlightthickness=1)
    layer_width_spinbox.delete(0, "end")
    layer_width_spinbox.insert(0, "4")
    layer_width_spinbox.grid(row=4, column=1, padx=10, pady=5)

    ctk.CTkButton(root, text="Start Training", command=start_training).grid(row=5, column=0, columnspan=2, pady=10)

    # Label to display training completion message
    training_complete_label = ctk.CTkLabel(root, text="Network training completed", font=("Arial", 16, "bold"), text_color="green")

    # Add canvas with scrollbars
    canvas_frame = ctk.CTkFrame(root)
    canvas_frame.grid(row=7, column=0, columnspan=2, padx=10, pady=10)

    canvas = ctk.CTkCanvas(canvas_frame, bg="white")
    h_scroll = ctk.CTkScrollbar(canvas_frame, orientation="horizontal", command=canvas.xview)
    v_scroll = ctk.CTkScrollbar(canvas_frame, orientation="vertical", command=canvas.yview)
    canvas.configure(xscrollcommand=h_scroll.set, yscrollcommand=v_scroll.set)

    h_scroll.pack(side="bottom", fill="x")
    v_scroll.pack(side="right", fill="y")
    canvas.pack(side="left", expand=True, fill="both")

    root.mainloop()

if __name__ == '__main__':
    train()
