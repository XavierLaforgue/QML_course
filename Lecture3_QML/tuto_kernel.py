"""
A simplistic example of using a quantum computer to evaluate a kernel.
Here, we use the SWAP test to measure the similarity between two input states,
followed by a classical SVM to classify the data.
"""

import numpy as np
from pennylane import numpy as pnp
import pennylane as qml
import matplotlib.pyplot as plt

N = 20 # number of train inputs
M = 20 # number of test inputs
relative_angle, scale = np.pi, 0.3

# Generate data. In real life, this should of course NOT be done like this,
# but rather cleanly in a separate module, and loaded properly.
# Here, data are rotation angles, sampled around two different means.
# This defines two categories, and the goal is then to classify data correctly.
# The relative angle between the two centroids controls the distance between the categories, and it is interesting to study the performance of the algorithm as a function of said angle.
def gen_data(N, relative_angle=relative_angle, scale=scale):
    X1 = np.random.normal(loc=0, scale=scale, size=N//2)
    X2 = np.random.normal(loc=relative_angle, scale=scale, size=N-len(X1))
    Y1 = np.zeros(len(X1))
    Y2 = np.ones(len(X2))
    return list(X1)+list(X2), list(Y1)+list(Y2)
X, Y = gen_data(N)

# Declare the device (simulator)
dev = qml.device("lightning.qubit", wires=3)

# Set up the quantum circuit (default interface = autograd)
@qml.qnode(dev)
def swap_test(angle1,angle2):

    # Note that with angle-encoded states, doing RY(angle1).RY^dag(angle2) would give the same result at a lower cost.
    qml.RY(angle1, wires=0)
    qml.RY(angle2, wires=1)

    # Swap the qubits 0 <-> 1
    qml.H(wires=2)
    qml.CSWAP(wires=[2,0,1])
    qml.H(wires=2)
    
    return qml.expval(qml.Z(2))

# Compute the Gram matrix, i.e. the matrix of kernels between the train inputs
K = np.eye(N,N) # NOTE this makes the system ill-conditioned, can I just remove it?
K = np.zeros((N,N))
for i in range(len(X)):
    angle1 = X[i]
    for j in range(i):
        angle2 = X[j]
        overlap = swap_test(angle1, angle2)
        K[i,j] = overlap
        K[j,i] = overlap

# Solve the linear system. NOTE: no regularisation here (i) to keep things simple (ii) it works fine without
weights = np.linalg.solve(K, Y)
print(f"Weights: {weights}")

# Prepare test data and compute the kernels between train and test inputs
X_test, Y_test = gen_data(M)
kernels = np.zeros((N,M))
for idx_test in range(M):
    new_angle = X_test[idx_test]
    for idx_train in range(N):
        kernels[idx_train,idx_test] = swap_test(X[i], new_angle)

# Compute the predicted labels for the test data
preds = weights.T @ kernels
print(f"Predicted labels: {preds}")

# Compute some performance metrics
mse = np.sum( (preds - Y_test)**2 ) / M
accuracy = 100 * sum(np.abs(preds - Y_test) < 1e-1) / M
print(f"MSE loss: {mse}")
print(f"Accuracy: {accuracy}%")

# Some plots
angles = np.linspace(0, 2*np.pi, 1000)
plt.plot(np.cos(angles), np.sin(angles), color="k")
# train dataset
plt.scatter(np.cos(X[:N//2]), np.sin(X[:N//2]), color="tab:blue", marker="o", label="Train, y=0")
plt.scatter(np.cos(X[N//2:]), np.sin(X[N//2:]), color="tab:orange", marker="s", label="Train, y=1")
# test dataset
plt.scatter(np.cos(X_test[:M//2]), np.sin(X_test[:M//2]), color="tab:green", marker="d", label="Test, y=0")
plt.scatter(np.cos(X_test[M//2:]), np.sin(X_test[M//2:]), color="tab:red", marker="*", label="Test, y=1")

plt.legend()
plt.savefig("kernel_classifier.pdf", bbox_inches="tight")
plt.show()
