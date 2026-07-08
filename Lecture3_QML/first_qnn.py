"""
A simplistic QML algorithm.
Here, we try to fit data generated as y=cos(x) using a single RY rotation and data reuploading.
Note that since the QNN can implement <Z(x,theta)>=cos(x) exactly, this will converge regardless of the amount of data.
"""

import numpy as np
from pennylane import numpy as pnp
import pennylane as qml
import matplotlib.pyplot as plt

# Try loading TeX for prettier plots. NOTE: this makes plotting _slow_ (e.g. ~1-2 seconds)
try:
    plt.rcParams.update({
    'text.usetex': True,
    'font.size': 14,
    })
except:
    pass

# Generate data. In real life, this should of course NOT be done like this,
# but rather cleanly in a separate module, and loaded properly.
def gen_data(N):
    X = np.random.uniform(size=N)
    Y = np.cos(X)
    #Y = np.sin(2*X)
    return X, Y
X, Y = gen_data(10)

# Declare the device (here a simulator)
dev = qml.device('lightning.qubit', wires=1)

# Set up the quantum circuit (default interface = autograd)
@qml.qnode(dev)
def circuit(params, x):
    # Awesome parametric ansatz using data reuploading
    qml.RY(params[0] * x + params[1], wires=0)
    return qml.expval(qml.Z(0))

# Cost function
def cost(param):
    loss = 0.0
    for x, y in zip(X, Y):
        loss += (circuit(param, x) - y)**2
    return loss / len(Y) 

print(cost([2.0,0.0]))

# Optimisation parameters
max_iterations = 100
conv_tol = 1e-5
learning_rate = 0.3

# PennyLane optimiser (gradient descent)
opt = qml.GradientDescentOptimizer(stepsize=learning_rate)

# Initial parameters
theta = pnp.array([1.23, 0.456], requires_grad=True)

# Store the cost and parameters
energy = [cost(theta)]
angles = [theta]

# Variational loop
for n in range(max_iterations):
    print(f'Iteration {n} (total {max_iterations})', end='\r')
    
    theta = opt.step(cost, theta)
 
    angles.append(theta)
    energy.append(cost(theta))

    conv = abs(energy[-1] - energy[-2])

    if conv <= conv_tol:
        break

print('\nFinal value of the cost = {:.8f}'.format(energy[-1]))
print(f'\nOptimal parameters {angles[-1]}')

##############################################################################
# Plot results

fig = plt.figure(figsize=(8, 3))

# Energy plot
ax1 = fig.add_subplot(121)
ax1.plot(range(n + 2), energy, 'go', linestyle='-')
ax1.set_xlabel('Optimisation step')
ax1.set_ylabel('Cost')
ax1.tick_params()

# Parameters plot
weights = [angle[0] for angle in angles]
biases  = [angle[1] for angle in angles]
ax2 = fig.add_subplot(122)
ax2.plot(range(n + 2), weights, 'ro', linestyle='-', label=r'$\theta_0=$weight')
ax2.plot(range(n + 2), biases,  'bo', linestyle='-', label=r'$\theta_1=$bias')
ax2.set_xlabel('Optimisation step')
ax2.set_ylabel(r'Gate parameters $\theta_i$ (rad)')
ax2.tick_params()

# Save the figure
plt.legend()
plt.subplots_adjust(wspace=0.3, bottom=0.2)
plt.savefig('plot-first-qnn.png', transparent=True, bbox_inches='tight', dpi=600)
plt.show()

