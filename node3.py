import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
from torchdiffeq import odeint
import seaborn
import os
class ODETestingModule:
    def __init__(self, model, mean, std):
        self.model = model
        self.mean = mean
        self.std = std
        self.model.eval()
    
    def normalize_state(self, state):
        """Normalize the state using stored mean and std"""
        return (state - self.mean) / self.std
    
    def denormalize_state(self, state):
        """Denormalize the state using stored mean and std"""
        return state * self.std + self.mean
    
    def model_derivative(self, state):
        """Get the derivative from the model"""
        # Ensure state is normalized before passing to model
        normalized_state = self.normalize_state(state)
        with torch.no_grad():
            # Use t=0 as we only need instantaneous derivative
            t = torch.tensor([0.0])
            derivative = self.model.ode_func(t, normalized_state)
            # Denormalize the derivative
            return derivative * self.std  # Only multiply by std for derivatives
    
    def rk4_step(self, state, dt):
        """Perform a single RK4 integration step"""
        k1 = self.model_derivative(state)
        k2 = self.model_derivative(state + k1 * dt / 2)
        k3 = self.model_derivative(state + k2 * dt / 2)
        k4 = self.model_derivative(state + k3 * dt)
        return state + (k1 + 2 * k2 + 2 * k3 + k4) * dt / 6
    
    def rk4_integrate(self, initial_state, t_span, steps=1000):
        """Integrate using RK4 method"""
        dt = t_span / steps
        states = []
        state = torch.tensor(initial_state, dtype=torch.float32)
        
        for _ in range(steps):
            state = self.rk4_step(state, dt)
            states.append(state.numpy())
        
        return np.array(states)

def test_and_visualize(model, mean, std, data, initial_state=None, t_span=25.0, steps=1000, save_path='plots'):

    import os
    import matplotlib.pyplot as plt
    
    # Create save directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    
    if initial_state is None:
        initial_state = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    
    # Initialize testing module
    tester = ODETestingModule(model, mean, std)
    
    # Generate predictions
    predicted_states = tester.rk4_integrate(initial_state, t_span, steps)

    t = np.linspace(0, t_span, steps)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot3D(predicted_states[:, 0], predicted_states[:, 1], predicted_states[:, 2], 
              'red', label='Predicted', linewidth=1)
    ax.plot3D(data['x'], data['y'], data['z'], 
              'blue', label='Actual', alpha=0.5, linewidth=1)
    ax.set_title('3D Space Trajectory')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.savefig(os.path.join(save_path, '3d_trajectory.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # X-Y projection
    plt.figure(figsize=(8, 6))
    plt.plot(predicted_states[:, 0], predicted_states[:, 1], 'red', label='Predicted', linewidth=1)
    plt.plot(data['x'], data['y'], 'blue', label='Actual', alpha=0.5, linewidth=1)
    plt.title('X-Y Projection')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.savefig(os.path.join(save_path, 'xy_projection.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Y-Z projection
    plt.figure(figsize=(8, 6))
    plt.plot(predicted_states[:, 1], predicted_states[:, 2], 'red', label='Predicted', linewidth=1)
    plt.plot(data['y'], data['z'], 'blue', label='Actual', alpha=0.5, linewidth=1)
    plt.title('Y-Z Projection')
    plt.xlabel('Y')
    plt.ylabel('Z')
    plt.legend()
    plt.savefig(os.path.join(save_path, 'yz_projection.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Time series plots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
    
    ax1.plot(t, predicted_states[:, 0], 'red', label='Predicted', linewidth=1)
    ax1.plot(t[:len(data['x'])], data['x'], 'blue', label='Actual', alpha=0.5, linewidth=1)
    ax1.set_title('X Time Series')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('X')
    ax1.legend()
    
    ax2.plot(t, predicted_states[:, 1], 'red', label='Predicted', linewidth=1)
    ax2.plot(t[:len(data['y'])], data['y'], 'blue', label='Actual', alpha=0.5, linewidth=1)
    ax2.set_title('Y Time Series')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Y')
    ax2.legend()
    
    ax3.plot(t, predicted_states[:, 2], 'red', label='Predicted', linewidth=1)
    ax3.plot(t[:len(data['z'])], data['z'], 'blue', label='Actual', alpha=0.5, linewidth=1)
    ax3.set_title('Z Time Series')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Z')
    ax3.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'time_series.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Calculate and save error metrics
    pred_length = min(len(predicted_states), len(data['x']))
    mse = {
        'x': np.mean((predicted_states[:pred_length, 0] - data['x'][:pred_length])**2),
        'y': np.mean((predicted_states[:pred_length, 1] - data['y'][:pred_length])**2),
        'z': np.mean((predicted_states[:pred_length, 2] - data['z'][:pred_length])**2)
    }
    
    # Save metrics to a text file
    with open(os.path.join(save_path, 'error_metrics.txt'), 'w') as f:
        f.write("Error Metrics (MSE):\n")
        f.write(f"X-coordinate MSE: {mse['x']:.6f}\n")
        f.write(f"Y-coordinate MSE: {mse['y']:.6f}\n")
        f.write(f"Z-coordinate MSE: {mse['z']:.6f}\n")
        f.write(f"Total MSE: {sum(mse.values())/3:.6f}\n")
    
    return predicted_states, mse

def load_data(file_path):
    data = np.load(file_path, allow_pickle=True)['simulations']
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
            continue
    return data, data_points
class ODEFunc(nn.Module):
    def __init__(self, hidden_dim):
        super(ODEFunc, self).__init__()
        # Deeper network with residual connections
        self.net = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3)
        )
        
        # Initialize weights using Kaiming initialization
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)
    
    def forward(self, t, x):
        return self.net(x)

class NeuralODE(nn.Module):
    def __init__(self):
        super(NeuralODE, self).__init__()
        self.ode_func = ODEFunc(hidden_dim=32)
    
    def forward(self, x, t_span):
        return odeint(self.ode_func, x, t_span, method='dopri5', rtol=1e-7, atol=1e-9)


def prepare_data(data_points):
    data_tensors = [torch.tensor(sim_data, dtype=torch.float32) for sim_data in data_points]
    

    all_data = torch.cat(data_tensors)
    mean = all_data.mean(dim=0)
    std = all_data.std(dim=0)
    

    normalized_tensors = [(tensor - mean) / std for tensor in data_tensors]
    

    X_train = [sim_data[:-1] for sim_data in normalized_tensors]
    y_train = [sim_data[1:] for sim_data in normalized_tensors]
    
    return torch.cat(X_train), torch.cat(y_train), mean, std

# Training function
def train_model(model, X_train, y_train, X_val, y_val, epochs=15000):
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=200, verbose=True)
    criterion = nn.MSELoss()
    
    batch_size = 128
    t_span = torch.linspace(0, 5, 200)  # Smaller time steps
    best_val_loss = float('inf')
    patience_counter = 0
    patience_limit = 10
    
    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(X_train.size(0))
        epoch_loss = 0
        
        for i in range(0, X_train.size(0), batch_size):
            indices = permutation[i:i + batch_size]
            batch_X, batch_y = X_train[indices], y_train[indices]
            
            optimizer.zero_grad()
            pred_y = model(batch_X, t_span)
            loss = criterion(pred_y[-1], batch_y)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            epoch_loss += loss.item()
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val, t_span)
            val_loss = criterion(val_pred[-1], y_val)
            
        scheduler.step(val_loss)
        
        if epoch % 1 == 0:
            print(f"Epoch {epoch}, Train Loss: {epoch_loss / (X_train.size(0) / batch_size):.6f}, Val Loss: {val_loss:.6f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience_limit:
            print(f"Early stopping triggered at epoch {epoch}")
            break
    
    return model


def main():
    # Load data
    data, data_points = load_data("/Users/josephmurray/Documents/PIML 597/lorenz_data.npz")
    

    X_train_full, y_train_full, mean, std = prepare_data(data_points)
    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, 
                                                      test_size=0.2, random_state=42)
    
    # Initialize and train model
    save_path = "/Users/josephmurray/Documents/PIML 597"
    model = NeuralODE()
    trained_model= train_model(model, X_train, y_train, X_val, y_val)
    torch.save(trained_model,os.path.join(save_path, 'trained_model.pth'))

    initial_state = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    #predicted_states = predict_trajectory(trained_model, initial_state, mean, std, t_span=25.0)
    
    # Plot results
    predicted_states, mse = test_and_visualize(model, mean, std, data[0], 
                                         initial_state=initial_state,
                                         t_span=25.0, 
                                         steps=10000)
    
    return trained_model, mean, std

if __name__ == "__main__":
    model, mean, std = main()
    #data, data_points = load_data("/Users/josephmurray/Documents/PIML 597/lorenz_data.npz")
    #model = torch.load('trained_model.pth')
    # Prepare and normalize data
    #X_train_full, y_train_full, mean, std = prepare_data(data_points)
    #initial_state = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    #predicted_states, mse = test_and_visualize(model, mean, std, data[0], 
    #                        initial_state=initial_state,
    #                        t_span=25, 
    #                        steps=1000)

    

# Prediction and Plotting
