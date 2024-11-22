import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torchdiffeq import odeint

# Load and process data
data = np.load("/Users/josephmurray/Documents/PIML 597/lorenz_data.npz", allow_pickle=True)['simulations']
data_points = []

# Process each simulation, ensuring data consistency
for idx, sim in enumerate(data):
    try:
        x = np.asarray(sim['x'], dtype=np.float32)
        y = np.asarray(sim['y'], dtype=np.float32)
        z = np.asarray(sim['z'], dtype=np.float32)
        simulation_data = np.column_stack([x, y, z])
        data_points.append(simulation_data)
    except ValueError as e:
        print(f"ValueError in simulation {idx}: {e}")
        continue  # Skip any problematic simulations

# Convert each simulation to PyTorch tensors
data_tensors = [torch.tensor(sim_data, dtype=torch.float32) for sim_data in data_points]

# Prepare training data: X_train and y_train for each simulation
X_train = [sim_data[:-1] for sim_data in data_tensors]
y_train = [sim_data[1:] for sim_data in data_tensors]

# Define NeuralODE model components
class ODEFunc(nn.Module):
    def __init__(self, hidden_dim):
        super(ODEFunc, self).__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()
    
    def forward(self, t, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)

class NeuralODE(nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super(NeuralODE, self).__init__()
        self.fc_in = nn.Linear(output_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.ode_func = ODEFunc(hidden_dim)
    
    def forward(self, x, t_span):
        x = self.fc_in(x)
        out = odeint(self.ode_func, x, t_span)
        return self.fc_out(out)

# Convert data for batch processing
X_train = torch.cat(X_train)
y_train = torch.cat(y_train)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Model, optimizer, and loss function
model = NeuralODE(hidden_dim=64, output_dim=3)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Training loop
epochs = 10000
batch_size = 64
t_step = torch.tensor([1.0])  # Single time step at t=1

for epoch in range(epochs):
    permutation = torch.randperm(X_train.size(0))
    epoch_loss = 0

    for i in range(0, X_train.size(0), batch_size):
        indices = permutation[i:i + batch_size]
        batch_X, batch_y = X_train[indices], y_train[indices]

        optimizer.zero_grad()
        
        # Forward pass with NeuralODE, using a single time step
        pred_y = model(batch_X, t_step)
        
        # Only take the prediction at the single time step
        loss = criterion(pred_y[-1], batch_y)
        epoch_loss += loss.item()
        
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch + 1}, Loss: {epoch_loss / (X_train.size(0) / batch_size)}")

# RK4 Method for Predictions

def rk4_method(initial_state, t_span, model, dt=0.01, steps=100):
    states = []
    state = torch.tensor(initial_state, dtype=torch.float32)
    for _ in range(steps):
        k1 = model(state.unsqueeze(0)).squeeze() * dt
        k2 = model((state + k1 / 2).unsqueeze(0)).squeeze() * dt
        k3 = model((state + k2 / 2).unsqueeze(0)).squeeze() * dt
        k4 = model((state + k3).unsqueeze(0)).squeeze() * dt
        state = state + (k1 + 2 * k2 + 2 * k3 + k4) / 6
        states.append(state.detach().numpy())
    return np.array(states)

# Prediction and Plotting
initial_state = np.array([1.0, 1.0, 1.0], dtype=np.float32)

predicted_states = rk4_method(initial_state, model, t_span=[0, 25], steps=1000)

# Load actual data for comparison
actual_data = data[0]
plt.figure(figsize=(10, 7))
plt.plot(predicted_states[:, 0], predicted_states[:, 1], label="Predicted", color="red")
plt.plot(actual_data['x'], actual_data['y'], label="Actual", color="blue", alpha=0.5)
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.show()
