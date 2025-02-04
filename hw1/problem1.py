import random as rand
import numpy as np
import matplotlib.pyplot as plt

def generate_data(n,d):
    X_train = np.random.randn(n, d)/np.sqrt(d)
    return X_train

def nearest_neigbor_distance(X_train, x):
    distances = np.linalg.norm(X_train - x, axis = 1)
    return np.min(distances)

def expected_distance(n, d):
    X_train = generate_data(n, d)
    distances = [nearest_neigbor_distance(X_train, np.random.randn(d)/np.sqrt(d)) for _ in range(100)]
    return np.mean(distances)

d_values = [2,4,6,8,10]
n_values = [100,200,500,1000,2000,5000]

expected_distances = np.zeros((len(d_values), len(n_values)))

for i, d in enumerate(d_values):
    for j, n in enumerate(n_values):
        expected_distances[i,j] = expected_distance(n, d)


plt.figure(figsize=(10, 6))
for i, d in enumerate(d_values):
    plt.plot(n_values, expected_distances[i], marker='o', label=f'd={d}')

plt.xscale("log") 
plt.xlabel("Number of training points (n)")
plt.ylabel("Expected nearest neighbor distance E[d(x)]")
plt.title("Expected Nearest Neighbor Distance vs. n for different d")
plt.legend()
plt.grid(True)
plt.show()    




