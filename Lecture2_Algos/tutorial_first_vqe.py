"""
A simplistic variational quantum algorithm.
Here, we optimise the single qubit state |psi(theta)> = RY(theta)|0> with respect to
the cost Hamiltonian X.
"""

# Somehow PennyLane begs me to use JAX now, so I oblige
from jax import numpy as np
import jax
jax.config.update("jax_platform_name", "cpu")
jax.config.update('jax_enable_x64', True)
import pennylane as qml
import optax

# Declare the device (here a simulator)
dev = qml.device("lightning.qubit", wires=1)

# Set up the quantum circuit
@qml.qnode(dev, interface="jax")
def circuit(param):
    qml.RY(param, 0)
    return qml.expval(qml.X(0))

# Set up the cost function
def cost_fn(param):
    return circuit(param)

# Set up the optimiser and store useful quantities
max_iterations = 100
conv_tol = 1e-06
opt = optax.sgd(learning_rate=0.4)
theta = np.array(0.) # initial angles
opt_state = opt.init(theta)

energy = [cost_fn(theta)] # store the values of the cost function
angle = [theta] # store the values of the optimised parameters

# Variational loop
for n in range(max_iterations):

    gradient = jax.grad(cost_fn)(theta)
    updates, opt_state = opt.update(gradient, opt_state)
    theta = optax.apply_updates(theta, updates)
    
    angle.append(theta)
    energy.append(cost_fn(theta))

    conv = np.abs(energy[-1] - energy[-2]) # assess convergence

    if n % 2 == 0:
        print(f"Step = {n},  Cost = {energy[-1]:.8f}")

    if conv <= conv_tol:
        break

print("\n" f"Final value of the ground-state energy = {energy[-1]:.8f}")
print("\n" f"Optimal value of the circuit parameter = {angle[-1]:.4f}")

##############################################################################
# Plot the value of the cost function, and the variational parameter theta
# as a function of the optimisation step.

import matplotlib.pyplot as plt

fig = plt.figure()
fig.set_figheight(5)
fig.set_figwidth(12)

# Exact minimum of the cost function
cost_exact = -1.0

# Add energy plot on column 1
ax1 = fig.add_subplot(121)
ax1.plot(range(n + 2), energy, "go", ls="dashed")
ax1.plot(range(n + 2), np.full(n + 2, cost_exact), color="red")
ax1.set_xlabel("Optimisation step", fontsize=13)
ax1.set_ylabel("Cost", fontsize=13)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Add angle plot on column 2
ax2 = fig.add_subplot(122)
ax2.plot(range(n + 2), angle, "go", ls="dashed")
ax2.set_xlabel("Optimisation step", fontsize=13)
ax2.set_ylabel("Gate parameter $\\theta$ (rad)", fontsize=13)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.subplots_adjust(wspace=0.3, bottom=0.2)
plt.show()

