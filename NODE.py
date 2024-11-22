import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
# Load data (use your generated Lorenz data)
from torchdiffeq import odeint
# Load data
data = np.load("/Users/josephmurray/Documents/PIML 597/lorenz_data.npz", allow_pickle=True)['simulations']
data_points = []

# Process each simulation and check for data type consistency
for idx, sim in enumerate(data):
    try:
        # Verify that all elements are numeric
        x = np.asarray(sim['x'], dtype=np.float32)
        y = np.asarray(sim['y'], dtype=np.float32)
        z = np.asarray(sim['z'], dtype=np.float32)

        # Stack x, y, z for each time step within this simulation
        simulation_data = np.column_stack([x, y, z])
        data_points.append(simulation_data)
    
    except ValueError as e:
        print(f"ValueError in simulation {idx}: {e}")
        continue  # Skip this simulation if there's an error

# Convert each simulation to a PyTorch tensor separately
if data_points:
    data_tensors = [torch.tensor(sim_data, dtype=torch.float32) for sim_data in data_points]

    # Example: Creating X_train and y_train for each simulation independently
    X_train = [sim_data[:-1] for sim_data in data_tensors]
    y_train = [sim_data[1:] for sim_data in data_tensors]
    
    print(f"Number of simulations: {len(data_tensors)}")
    for i, (X, y) in enumerate(zip(X_train, y_train)):
        print(f"Simulation {i}: X_train shape={X.shape}, y_train shape={y.shape}")
else:
    print("No valid data points found.")

def rk4_method(initial_state, t_span, model, dt=0.01, steps=5):
    states = []
    state = torch.tensor(initial_state, dtype=torch.float32)
    for t in np.arange(t_span[0], t_span[1], dt)[:steps]:
        k1 = model(state.unsqueeze(0)).squeeze() * dt
        k2 = model((state + k1 / 2).unsqueeze(0)).squeeze() * dt
        k3 = model((state + k2 / 2).unsqueeze(0)).squeeze() * dt
        k4 = model((state + k3).unsqueeze(0)).squeeze() * dt
        state = state + (k1 + 2 * k2 + 2 * k3 + k4) / 6
        states.append(state.detach().numpy())
    return np.array(states)
# Define the neural network model
class DerivativeNN(nn.Module):
    def __init__(self):
        super(DerivativeNN, self).__init__()
        self.fc1 = nn.Linear(3, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 3)
        
    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = torch.relu(self.fc4(x))
        return self.fc5(x)
class ODEFunc(nn.Module):
    def __init__(self, hidden_dim):
        super(ODEFunc, self).__init__()
        # Define a small MLP to represent the dynamics
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()
    
    def forward(self, t, x):
        # x is the current state of the system
        out = self.relu(self.fc1(x))
        out = self.fc2(out)
        return out

# Step 2: Define the Neural ODE Model
class NeuralODE(nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super(NeuralODE, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.fc_in = nn.Linear(output_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.ode_func = ODEFunc(hidden_dim)
    
    def forward(self, x, t_span):
        # Encode input into the hidden state
        x = self.fc_in(x)
        
        # Solve ODE
        out = odeint(self.ode_func, x, t_span)
        
        # Decode hidden state back to output
        out = self.fc_out(out)
        return out

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
X_train, y_train = torch.tensor(X_train), torch.tensor(y_train)
X_val, y_val = torch.tensor(X_val), torch.tensor(y_val)


# Instantiate the model and optimizer
model = NeuralODE(hidden_dim=64, output_dim=X_train.size(1))
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Prepare training data

# Training loop
epochs = 10000
batch_size = 64
patience = 500  # Number of epochs to wait before stopping if no improvement
min_delta = 0.001  # Minimum change in loss to qualify as an improvement
dt = 0.01
steps = 5
# Early stopping variables
best_loss = float('inf')
patience_counter = 0
t_span = torch.linspace(0, 1, 100)  # Integrate from time 0 to 1 over 10 steps

for epoch in range(epochs):
    permutation = torch.randperm(X_train.size(0))
    epoch_loss = 0

    for i in range(0, X_train.size(0), batch_size):
        indices = permutation[i:i + batch_size]
        batch_X, batch_y = X_train[indices], y_train[indices]

        optimizer.zero_grad()
        pred_y = model(batch_X, t_span)  # Forward pass
        loss = criterion(pred_y[-1], batch_y)  # MSE Loss
        epoch_loss += loss.item()
        loss.backward()  # Backprop
        optimizer.step()  # Update

    print(f"Epoch {epoch + 1}, Loss: {epoch_loss / (X_train.size(0) / batch_size)}")


# Euler Method for prediction


# Predict trajectory
initial_state = np.array([1.0, 1.0, 1.0], dtype=np.float32)
predicted_states = rk4_method(initial_state, [0, 25], model,steps= 1000)

# Plot the predicted vs actual Lorenz attractor
actual_data = np.load("lorenz_data.npz", allow_pickle=True)['simulations'][0]
plt.figure(figsize=(10, 7))
plt.plot(predicted_states[:, 0], predicted_states[:, 1], label="Predicted", color="red")
plt.plot(actual_data['x'], actual_data['y'], label="Actual", color="blue", alpha=0.5)
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.show()
