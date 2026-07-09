'''
A variational parity classifier, using the datasets from https://pennylane.ai/qml/demos/tutorial_variational_classifier
'''

import os
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
import matplotlib.pyplot as plt

# Load the data
data = np.loadtxt(os.getcwd() + '/parity_train.txt', dtype=int)
X_train = np.array(data[:, :-1])
Y_train = np.array(data[:, -1])
Y_train = Y_train * 2 - 1  # shift label from {0, 1} to {-1, 1}

data = np.loadtxt(os.getcwd() + '/parity_test.txt', dtype=int)
X_test = np.array(data[:, :-1])
Y_test = np.array(data[:, -1])
Y_test = Y_test * 2 - 1  # shift label from {0, 1} to {-1, 1}

# Declare the circuit and device
nb_wires = 4
dev = qml.device('default.qubit', wires=nb_wires)


# TODO: verify that the parity could also be checked without the rotation
# gates and just using the first three CNOT gates or with all cnots and some
# rotations

# Basis encoding
def basis_encoding(x):
    ''' Encode a binary vector x using basis encoding. '''

    assert (len(x) <= nb_wires), f'Cannot encode a vector of length {len(x)}\
        on {nb_wires}.'

    for wire, x_i in enumerate(x):
        if x_i == 1:
            qml.X(wires=wire)


# Parity classifier circuit
@qml.qnode(dev)
def parity_classifier(params, x):
    # First step: encode the data into the register
    basis_encoding(x)

    # Second step: variational ansatz. Note that the order is not the same as
    # in the figure, but this doesn't change anything. Students can take it
    # as an exercice to reorder the indices :)
    nb_params_per_gate = 3
    nb_parameterized_gates = 8
    nb_params = nb_params_per_gate * nb_parameterized_gates
    assert nb_params == len(params), f"Unexpected number of parameters.\
        Expected {nb_params} but received {len(params)}"
    for wire in range(nb_wires):
        qml.RZ(params[(nb_wires-1) * wire + 0], wires=wire)
        qml.RY(params[(nb_wires-1) * wire + 1], wires=wire)
        qml.RZ(params[(nb_wires-1) * wire + 2], wires=wire)
    for wire in range(nb_wires-1):
        qml.CNOT([wire, wire+1])
    qml.CNOT([nb_wires-1, 0])
    for wire in range(nb_wires):
        start_idx = (nb_wires-1) * wire + nb_wires * nb_params_per_gate
        qml.RZ(params[start_idx + 0], wires=wire)
        qml.RY(params[start_idx + 1], wires=wire)
        qml.RZ(params[start_idx + 2], wires=wire)

    return qml.expval(qml.Z(0))


# Cost function
def cost(params, X, Y):
    '''Calculate Mean Square Error (MSE) loss.'''
    loss = 0.0
    for x, y in zip(X, Y):
        loss += (parity_classifier(params, x) - y)**2
    return loss / len(Y)


# Optimisation parameters
np.random.seed(1)  # the seed is a hyperparameter. Here I mainly fix it to a
# value that makes convergence fast
max_iterations = 100
conv_tol = 1e-3
learning_rate = 0.4

# PennyLane optimiser (gradient descent)
opt = qml.GradientDescentOptimizer(stepsize=learning_rate)

# Initial parameter
params = pnp.array(2 * np.pi * (np.random.rand(24) - 0.5), requires_grad=True)

# Storage
train_loss = []
test_loss = []
params_hist = [params]  # TODO check if I don't neep .copy() here

# Variational loop
iteration = 0
for iteration in range(max_iterations):
    print(f'Iteration {iteration} (with cut-off: {max_iterations})', end='\r')

    params = opt.step(cost, params, X=X_train, Y=Y_train)

    train_loss.append(cost(params, X_train, Y_train))
    test_loss.append(cost(params, X_test, Y_test))
    params_hist.append(params)

    if iteration > 0:
        conv = abs(train_loss[-1] - train_loss[-2])
        if conv <= conv_tol:
            break
print(f'Last test loss: {test_loss[-1]}.')
print(f'Last train loss: {train_loss[-1]}.')

plt.plot(range(iteration+1), train_loss, label='Train loss')
plt.plot(range(iteration+1), test_loss, label='Test loss')
plt.legend()
plt.show()
