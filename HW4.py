import numpy as np
import matplotlib
import time
from matplotlib import pyplot as plt
import random
class NeuralNetwork:
    def sigmoid(Z):
        #print("Z_input=",Z)
        Z = 1/(1+np.exp(-Z))
        #print("Z =",Z)
        return Z

    def relu(Z):
        return np.maximum(0,Z)

    def sigmoid_backward(dA, Z):
        sig = NeuralNetwork.sigmoid(Z)
        return dA * sig * (1 - sig)

    def relu_backward(dA, Z):
        dZ = np.array(dA, copy = True)
        #print(dZ.shape)
        #print(Z.shape)
        dZ[Z <= 0] = 0
        return dZ
    def target_function(x, y):
        return np.sin(2 * np.pi * x * y) + 2 * x * y ** 2

    def set_layers(nn):
        np.random.seed(50)
        numb_layers = len(nn)
        params_values = {}
        for idx,layer in enumerate(nn):
            layer_idx = idx +1
            layer_input_size = layer["input_dim"]
            layer_output_size = layer["output_dim"]
            params_values['W' + str(layer_idx)] = np.random.randn(layer_input_size,
                layer_output_size) * 0.1
            params_values['b' + str(layer_idx)] = np.random.randn(layer_output_size,
                1) * 0.1

        return params_values

    def forward_pass(x,nn,params_values):
        num_layers = len(nn)
        #print("Input:",x)
        x = np.array(x).reshape(-1, 1)
        #x = np.transpose(x)
        
        for idx,num_layers in enumerate(nn):
            layer_idx = idx+1
            val_layer = np.dot(x,params_values['W'+ str(layer_idx)])+params_values['b' + str(layer_idx)]
            activation = NeuralNetwork.relu(val_layer)
            x = activation
        output = x
        #print("Output:",output)
        return output
    def loss1D(output, true):
        mse_loss = np.mean((true - output) ** 2)
        return mse_loss

    def accuracy(output, true):
        correct_predictions = np.sum(output == true)  # Count matching elements
        accuracy_percentage = (correct_predictions / len(true)) * 100  # Convert to percentage
        return accuracy_percentage

# Function Definitions
def target_function(x, y):
    return np.sin(2 * np.pi * x * y) + 2 * x * y ** 2
def generate_data(n_samples):
    x = np.random.uniform(0, 1, n_samples)
    y = np.random.uniform(0, 1, n_samples)
    z = target_function(x, y)
    X = np.array((x, y))
    y_true = z.reshape(-1, 1)

    return X, y_true
def single_layer_backward_propagation(dA_curr, W_curr, b_curr, Z_curr, A_prev, activation="relu"):
    #print(A_prev.shape[1])
    m = A_prev.shape[1]
    
    if activation == "relu":
        backward_activation_func = NeuralNetwork.relu_backward
    elif activation == "sigmoid":
        backward_activation_func = NeuralNetwork.sigmoid_backward
    else:
        raise Exception('Non-supported activation function')
    
    dZ_curr = backward_activation_func(dA_curr, Z_curr)
    dW_curr = np.dot(A_prev.T,dZ_curr)
    db_curr = np.sum(dZ_curr.T, axis=0, keepdims=True)
    dA_prev = np.dot(dZ_curr,W_curr.T)

    return dA_prev, dW_curr, db_curr
def single_layer_forward_propagation(A_prev, W_curr, b_curr, activation="relu"):
    #print("Input Layer Size:", A_prev.shape)
    #print("Weight Matrix size:",W_curr.shape)
    #print("Bias Matrix size:",b_curr.shape)
    Z_curr = np.dot(A_prev,W_curr)
    #print("Weight output:",Z_curr)
    Z_curr =Z_curr + np.transpose(b_curr)
    #print("Output Size:",Z_curr.shape)
    
    if activation == "relu":
        activation_func = NeuralNetwork.relu
    elif activation == "sigmoid":
        activation_func = NeuralNetwork.sigmoid
    else:
        raise Exception('Non-supported activation function')
    #print("Output:",activation_func(Z_curr))
    return activation_func(Z_curr), Z_curr
def full_forward_propagation(X, params_values, nn_architecture):
    memory = {}
    A_curr = X
    #print("A_initial",X)
    
    for idx, layer in enumerate(nn_architecture):
        layer_idx = idx + 1
        A_prev = A_curr
        #print(f"Layer {idx+1}: Input Dim: {layer['input_dim']}, Output Dim: {layer['output_dim']}, Activation: {layer['activation']}")
        activ_function_curr = layer["activation"]
        W_curr = params_values["W" + str(layer_idx)]
        #print("W_size", W_curr.size)
        b_curr = params_values["b" + str(layer_idx)]
        
        A_curr, Z_curr = single_layer_forward_propagation(A_prev, W_curr, b_curr, activ_function_curr)
        
        memory["A" + str(idx)] = A_prev
        memory["Z" + str(layer_idx)] = Z_curr
    #print(memory)
    #print("Final Output= ",A_curr)
    return A_curr, memory
def full_backward_propagation(Y_hat, Y, memory, params_values, nn_architecture):
    grads_values = {}
    m = Y.shape
    
   
    dA_prev = -(Y_hat-Y)
    for layer_idx_prev, layer in reversed(list(enumerate(nn_architecture))):
        layer_idx_curr = layer_idx_prev +1
        activ_function_curr = layer["activation"]
        
        dA_curr = dA_prev
        
        A_prev = memory["A" + str(layer_idx_prev)]
        Z_curr = memory["Z" + str(layer_idx_curr)]
        W_curr = params_values["W" + str(layer_idx_curr)]
        b_curr = params_values["b" + str(layer_idx_curr)]
        
        dA_prev, dW_curr, db_curr = single_layer_backward_propagation(
            dA_curr, W_curr, b_curr, Z_curr, A_prev, activ_function_curr)
        
        grads_values["dW" + str(layer_idx_curr)] = dW_curr
        grads_values["db" + str(layer_idx_curr)] = db_curr
    
    return grads_values
def forward_auto_diff(X, params_values, nn_architecture):
    memory = {}
    A_curr = X  # Initial input
    tangent_curr = np.ones_like(X)  # Start gradient tracking

    for layer_idx, layer in enumerate(nn_architecture):
        layer_idx_curr = layer_idx + 1
        W = params_values["W" + str(layer_idx_curr)]
        b = params_values["b" + str(layer_idx_curr)]
        activation = layer["activation"]
        #print("Weight Mat:",W.size)
        #print("B Size", b.size)
        #print("Tan size",tangent_curr.size)
        # Forward pass
        Z_curr = np.matmul(tangent_curr,W) + b.T
        if activation == "relu":
            dZ = NeuralNetwork.relu(Z_curr)
        elif activation == "sigmoid":
            A_curr, dZ = NeuralNetwork.sigmoid(Z_curr)
        else:
            raise Exception("Non-supported activation function")

        # Compute tangent (gradient of A_curr with respect to inputs)
        tangent_curr = dZ * (np.matmul(tangent_curr,W))
        #print("Updated Tan", tangent_curr.size)
        # Store values
        memory["A" + str(layer_idx_curr)] = A_curr
        memory["Z" + str(layer_idx_curr)] = Z_curr

    return A_curr, tangent_curr
def update(params_values, grads_values, nn_architecture, learning_rate):
    for layer_idx, layer in enumerate(nn_architecture):
        # Update weights and biases using the gradients corresponding to the random index
        params_values["W" + str(layer_idx+1)] -= learning_rate * grads_values["dW" + str(layer_idx+1)]      
        params_values["b" + str(layer_idx+1)] -= learning_rate * grads_values["db" + str(layer_idx+1)]

    return params_values
def train(target_function, nn_architecture, epochs, learning_rate, batch_size=32, verbose=True):
    params_values = NeuralNetwork.set_layers(nn_architecture)
    cost_history = []
    best_cost = float('inf')
    best_params = None
    patience = 50000
    patience_counter = 0
    
    # Learning rate scheduling
    initial_lr = learning_rate
    
    for epoch in range(epochs):
        epoch_costs = []
        
        for batch in range(batch_size):
            # Generate random points
            x = np.random.rand()
            y = np.random.rand()
            Y = target_function(x, y)
            X = np.array([x, y]).reshape(1, 2)
            
            # Forward pass
            Y_hat, cache = full_forward_propagation(X, params_values, nn_architecture)
            
            # Reshape Y to match Y_hat dimensions
            Y = np.array([[Y]])
            
            # Calculate loss
            cost = NeuralNetwork.loss1D(Y_hat, Y)
            epoch_costs.append(cost)
            
            # Backward pass
            grads_values = full_backward_propagation(Y_hat, Y, cache, params_values, nn_architecture)
            
            # Gradient clipping to prevent exploding gradients
            # Update parameters with current learning rate
            params_values = update(params_values, grads_values, nn_architecture, learning_rate)
        
        # Calculate average cost for the epoch
        avg_cost = np.mean(epoch_costs)
        cost_history.append(avg_cost)
        
        # Learning rate decay
        #learning_rate = initial_lr / (1 + epoch * 0.001)
        
        # Early stopping check
        if avg_cost < best_cost:
            best_cost = avg_cost
            best_params = {key: value.copy() for key, value in params_values.items()}
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Print progress
        if verbose and (epoch % 100 == 0 or epoch == epochs - 1):
            print(f"Epoch {epoch}/{epochs}, Loss: {avg_cost:.6f}, Learning Rate: {learning_rate:.6f}")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch}")
            break
    
    # Return the best parameters found during training
    return best_params, cost_history
nn_architecture= [
    {"input_dim": 2, "output_dim":10, "activation": "relu"},
    {"input_dim": 10, "output_dim":10, "activation":"relu"},
    {"input_dim": 10, "output_dim":10, "activation":"relu"},
    {"input_dim": 10, "output_dim":10, "activation":"relu"},
    {"input_dim": 10, "output_dim":1, "activation":"relu"},
]
def compare_gradient_computations(nn_architecture, params_values, num_tests=100):
    
    metrics = {
        'forward_times': [],
        'backward_times': [],
        'gradient_differences': [],
        'forward_gradients': [],
        'backward_gradients': []
    }
    
    for _ in range(num_tests):
        # Generate random test point
        x = np.random.rand()
        y = np.random.rand()
        X = np.array([x, y]).reshape(1, 2)
        Y = target_function(x, y)
        
        # Time forward autodiff
        start_time = time.time()
        _, forward_gradients = forward_auto_diff(X, params_values, nn_architecture)
        forward_time = time.time() - start_time
        metrics['forward_times'].append(forward_time)
        
        # Time backward propagation
        start_time = time.time()
        Y_hat, memory = full_forward_propagation(X, params_values, nn_architecture)
        grads_values = full_backward_propagation(Y_hat, Y, memory, params_values, nn_architecture)
        backward_time = time.time() - start_time
        metrics['backward_times'].append(backward_time)
        
        # Compute gradient norms
        forward_norm = np.linalg.norm(forward_gradients)
        backward_norm = np.sum([np.linalg.norm(grads_values["dW" + str(layer_idx + 1)]) 
                              for layer_idx in range(len(nn_architecture))])
        
        # Store gradients and their difference
        metrics['forward_gradients'].append(forward_norm)
        metrics['backward_gradients'].append(backward_norm)
        metrics['gradient_differences'].append(abs(forward_norm - backward_norm))
    
    # Compute summary statistics
    summary = {
        'avg_forward_time': np.mean(metrics['forward_times']),
        'avg_backward_time': np.mean(metrics['backward_times']),
        'std_forward_time': np.std(metrics['forward_times']),
        'std_backward_time': np.std(metrics['backward_times']),
        'avg_gradient_diff': np.mean(metrics['gradient_differences']),
        'max_gradient_diff': np.max(metrics['gradient_differences']),
        'min_gradient_diff': np.min(metrics['gradient_differences'])
    }
    
    # Print results
    print("\nGradient Computation Comparison Results:")
    print("-----------------------------------------")
    print(f"Average Forward Time: {summary['avg_forward_time']:.6f} ± {summary['std_forward_time']:.6f} seconds")
    print(f"Average Backward Time: {summary['avg_backward_time']:.6f} ± {summary['std_backward_time']:.6f} seconds")
    print(f"\nGradient Difference Statistics:")
    print(f"Average Difference: {summary['avg_gradient_diff']:.6f}")
    print(f"Maximum Difference: {summary['max_gradient_diff']:.6f}")
    print(f"Minimum Difference: {summary['min_gradient_diff']:.6f}")
    
    # Calculate speed comparison
    speedup = summary['avg_backward_time'] / summary['avg_forward_time']
    print(f"\nSpeed Comparison:")
    print(f"Forward method is {speedup:.2f}x {'faster' if speedup > 1 else 'slower'} than backward method")
    
    return metrics, summary
def test_neural_network(nn_architecture, trained_params, num_test_points=1000, plot_results=True):
    """
    Test the trained neural network and visualize its performance.
    
    Args:
        nn_architecture: Neural network architecture configuration
        trained_params: Trained network parameters
        num_test_points: Number of test points to evaluate
        plot_results: Whether to generate visualization plots
    
    Returns:
        dict: Dictionary containing test metrics and results
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    # Generate test points
    x_test = np.random.rand(num_test_points)
    y_test = np.random.rand(num_test_points)
    
    # Arrays to store results
    predictions = []
    true_values = []
    errors = []
    
    # Test the network
    for i in range(num_test_points):
        # Get true value
        true_value = target_function(x_test[i], y_test[i])
        true_values.append(true_value)
        
        # Get prediction
        X = np.array([x_test[i], y_test[i]]).reshape(1, 2)
        prediction, _ = full_forward_propagation(X, trained_params, nn_architecture)
        predictions.append(prediction[0][0])
        
        # Calculate error
        error = abs(prediction[0][0] - true_value)
        errors.append(error)
    
    # Calculate metrics
    mean_error = np.mean(errors)
    max_error = np.max(errors)
    min_error = np.min(errors)
    std_error = np.std(errors)
    mse = np.mean(np.square(errors))
    
    # Print metrics
    print("\nNeural Network Test Results:")
    print("-" * 30)
    print(f"Number of test points: {num_test_points}")
    print(f"Mean Absolute Error: {mean_error:.6f}")
    print(f"Max Error: {max_error:.6f}")
    print(f"Min Error: {min_error:.6f}")
    print(f"Standard Deviation of Error: {std_error:.6f}")
    print(f"Mean Squared Error: {mse:.6f}")
    
    if plot_results:
        # Create figure with subplots
        fig = plt.figure(figsize=(15, 5))
        
        # Plot 1: True vs Predicted Surface
        ax1 = fig.add_subplot(131, projection='3d')
        ax1.scatter(x_test, y_test, true_values, c='blue', label='True Values', alpha=0.5)
        ax1.scatter(x_test, y_test, predictions, c='red', label='Predictions', alpha=0.5)
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.set_title('True vs Predicted Values')
        ax1.legend()
        
        # Plot 2: Error Distribution
        ax2 = fig.add_subplot(132)
        ax2.hist(errors, bins=50, color='green', alpha=0.7)
        ax2.set_xlabel('Absolute Error')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Error Distribution')
        
        # Plot 3: Predicted vs True Values Scatter
        ax3 = fig.add_subplot(133)
        ax3.scatter(true_values, predictions, alpha=0.5)
        ax3.plot([min(true_values), max(true_values)], 
                 [min(true_values), max(true_values)], 
                 'r--', label='Perfect Prediction')
        ax3.set_xlabel('True Values')
        ax3.set_ylabel('Predicted Values')
        ax3.set_title('Predicted vs True Values')
        ax3.legend()
        
        plt.tight_layout()
        plt.show()
    
    # Create results dictionary
    results = {
        'predictions': predictions,
        'true_values': true_values,
        'errors': errors,
        'metrics': {
            'mean_error': mean_error,
            'max_error': max_error,
            'min_error': min_error,
            'std_error': std_error,
            'mse': mse
        },
        'test_points': {
            'x': x_test,
            'y': y_test
        }
    }
    
    return results
def visualize_training(cost_history):
    """
    Visualize the training progress.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(cost_history)
    plt.title('Training Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.grid(True)
    plt.show()
# Part A
params_values= NeuralNetwork.set_layers(nn_architecture)
x = np.random.rand()
y = np.random.rand()
x_true = target_function(x,y)
#jacobian_forward, grads_reverse = compute_gradients(x_true, nn_architecture, params_values)


X_input = [x,y]
#output = NeuralNetwork.forward_pass(X_input,nn_architecture,params_values)
#print("Inputs:", X_input)
#print("Forward pass outputs:", output)
#print("True output:",x_true)
#metrics,summary = compare_gradient_computations(nn_architecture,params_values,1000000)
#test_forward_backward_consistency(nn_architecture,params_values,x,y)
NN_trained,costhistory = train(target_function,nn_architecture,epochs=10000000, learning_rate = .01, batch_size=16, verbose=True)
visualize_training(costhistory)
results = test_neural_network(nn_architecture,NN_trained,10000,True)

## Part B

