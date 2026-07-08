'''
A variational parity classifier, using the datasets from https://pennylane.ai/qml/demos/tutorial_variational_classifier

Author: Yann Beaujeault-Taudière
© 2025-2026 To be defined
'''

import os
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
import matplotlib.pyplot as plt

# Load the data
data = np.loadtxt(os.getcwd() + "/parity_train.txt", dtype=int)
X_train = np.array(data[:, :-1])
Y_train = np.array(data[:, -1])
Y_train = Y_train * 2 - 1  # shift label from {0, 1} to {-1, 1}

data = np.loadtxt(os.getcwd() + "/parity_test.txt", dtype=int)
X_test = np.array(data[:, :-1])
Y_test = np.array(data[:, -1])
Y_test = Y_test * 2 - 1  # shift label from {0, 1} to {-1, 1}

# Declare the circuit and device
nb_wires = 4
dev = qml.device("default.qubit", wires=nb_wires)

# Basis encoding
def basis_encoding(x):
    ''' Encode a binary vector x using basis encoding. '''
    
    assert(len(x) <= nb_wires), f'Cannot encode a vector of length {len(x)} on {nb_qubits}.'
    
    for i in range(len(x)):
        if x[i] == 1:
            qml.X(wires=i)


# Parity classifier circuit
@qml.qnode(dev)
def parity_classifier(params, x):
    # First step: encode the data into the register
    basis_encoding(x)

    # Second step: variational ansatz. Note that the order is not the same as in the figure, but this doesn't change anything. Students can take it as an exercice to reorder the indices :)
    for wire in range(nb_wires):
        qml.RZ(params[6*wire]  , wires=wire)
        qml.RY(params[6*wire+1], wires=wire)
        qml.RZ(params[6*wire+2], wires=wire)
    for wire in range(nb_wires-1):
        qml.CNOT([wire, wire+1])
    qml.CNOT([nb_wires-1, 0])
    for wire in range(nb_wires):
        qml.RZ(params[6*wire+3]  , wires=wire)
        qml.RY(params[6*wire+4], wires=wire)
        qml.RZ(params[6*wire+5], wires=wire)

    return qml.expval(qml.Z(0))


# Cost function
def cost(params, X, Y):
    loss = 0.0
    for x, y in zip(X, Y):
        loss += (parity_classifier(params, x) - y)**2
    return loss / len(Y)


# Optimisation parameters
np.random.seed(1) # the seed is a hyperparameter. Here I mainly fix it to a value that makes convergence fast
max_iterations = 100
conv_tol = 1e-3
learning_rate = 0.4

# PennyLane optimiser (gradient descent)
opt = qml.GradientDescentOptimizer(stepsize=learning_rate)

# Initial parameter
params = pnp.array(2 * np.pi * (np.random.rand(24) - 0.5), requires_grad=True)

# Storage
train_loss = []
test_loss  = []
params_hist = [params] # TODO check if I don't neep .copy() here

# Variational loop
for iteration in range(max_iterations):
    print(f"Iteration {iteration} (total {max_iterations})", end="\r")

    params = opt.step(cost, params, X=X_train, Y=Y_train)

    train_loss.append(cost(params, X_train, Y_train))
    test_loss.append(cost(params, X_test, Y_test))
    params_hist.append(params)

    if iteration > 0:
        conv = abs(train_loss[-1] - train_loss[-2])
        if conv <= conv_tol:
            break

plt.plot(range(iteration+1), train_loss, label='Train loss')
plt.plot(range(iteration+1), test_loss, label='Test loss')
plt.legend()
plt.show()

