# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 08:55:11 2024

@author: lrdvi
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import pandas as pd
import random
# Define the Lorenz system equations
def Dynamics_lorenz_system(t, state, sigma, rho, beta):
    x, y, z = state
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return [dxdt, dydt, dzdt]

# Define the Lorenz system parameters
sigma = 10.0
rho = 28.0
beta = 8.0 / 3.0

# Set initial conditions
state_i = [1.0, 1.0, 1.0]

# Define the time points for integration
ti = 0
tf = 25
tspan = [ti, tf]
t_eval = np.linspace(ti, tf, 1000)  #

# Solve the Lorenz system using odeint
sol = solve_ivp(Dynamics_lorenz_system, tspan, state_i, method='RK45', t_eval=t_eval,args=(sigma, rho, beta))

# Extract the x, y, z components of the solution
x, y, z = sol.y

# Plot the Lorenz attractor
#fig = plt.figure(figsize=(10, 7))
#ax = fig.add_subplot(111, projection='3d')
#ax.plot(x, y, z, color='blue', lw=0.5)
#ax.set_title("Lorenz Attractor")
#ax.set_xlabel("X")
#ax.set_ylabel("Y")
#ax.set_zlabel("Z")
#plt.show()
num_iterations = 500
print(range(num_iterations))
sigma = 10.0
rho = 28.0
beta = 8.0 / 3.0
all_simulations = []
for i in range(num_iterations):
    state_i = [np.random.uniform(-20, 20), np.random.uniform(-20, 20), np.random.uniform(0, 50)]
    
    # Solve the Lorenz system using solve_ivp
    sol = solve_ivp(Dynamics_lorenz_system, [ti, tf], state_i, method='RK45', t_eval=t_eval, args=(sigma, rho, beta))
    
    # Store results in a dictionary for each simulation
    sim_data = {
        'time': sol.t,
        'x': sol.y[0],
        'y': sol.y[1],
        'z': sol.y[2]
    }
    all_simulations.append(sim_data)
# Save the generated data
print(len(all_simulations))
np.savez("lorenz_data.npz", simulations = all_simulations)
data = pd.DataFrame(all_simulations[0])

# Save the data to a CSV file
data.to_csv('lorenz_data.csv', index=False)

#%% loading data
'''
data = np.load("lorenz_data_ivp.npz")
x = data['x']
y = data['y']
z = data['z']
t = data['t']

data = pd.read_csv('lorenz_data.csv')
t = data['time'].values
x = data['x'].values
y = data['y'].values
z = data['z'].values
'''