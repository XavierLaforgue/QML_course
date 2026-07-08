"""
A simplistic variational quantum algorithm.
Here, we optimise the single qubit state |psi(theta)> = RY(theta)|0>
with respect to the cost Hamiltonian X.
"""

import numpy as np
import pennylane as qml
import matplotlib.pyplot as plt
from pennylane import numpy as pnp

# Declare the device (simulator)
dev = qml.device("lightning.qubit", wires=1)

# Set up the quantum circuit (default interface = autograd)
@qml.qnode(dev)
def circuit(param):
    qml.RY(param, wires=0)
    return qml.expval(qml.X(0))

# Cost function
def cost_fn(param):
    return circuit(param)

# Optimisation parameters
max_iterations = 100
conv_tol = 1e-6
learning_rate = 0.4

# PennyLane optimiser (gradient descent)
opt = qml.GradientDescentOptimizer(stepsize=learning_rate)

# Initial parameter
theta = pnp.array([1.2], requires_grad=True)


# Storage
energy = [cost_fn(theta)]
angle = [theta]

# Variational loop
for n in range(max_iterations):

    theta = opt.step(cost_fn, theta)

    angle.append(theta)
    energy.append(cost_fn(theta))

    conv = abs(energy[-1] - energy[-2])

    if n % 2 == 0:
        #print(f"Step = {n},  Cost = {energy[-1]:.8f}")

        if conv <= conv_tol:
            break

#print("\nFinal value of the ground-state energy = {:.8f}".format(energy[-1]))
#print("Optimal value of the circuit parameter = {:.4f}".format(angle[-1]))

##############################################################################
# Plot results

fig = plt.figure(figsize=(12, 5))

# Exact minimum
cost_exact = -1.0

# Energy plot
ax1 = fig.add_subplot(121)
ax1.plot(range(n + 2), energy, "go", ls="dashed")
ax1.plot(range(n + 2), np.full(n + 2, cost_exact), color="red")
ax1.set_xlabel("Optimisation step", fontsize=13)
ax1.set_ylabel("Cost", fontsize=13)
ax1.tick_params(labelsize=12)

# Parameter plot
ax2 = fig.add_subplot(122)
ax2.plot(range(n + 2), angle, "go", ls="dashed")
ax2.set_xlabel("Optimisation step", fontsize=13)
ax2.set_ylabel(r"Gate parameter $\theta$ (rad)", fontsize=13)
ax2.tick_params(labelsize=12)

plt.subplots_adjust(wspace=0.3, bottom=0.2)
plt.show()

