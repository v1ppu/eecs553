import random as rand
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

def generate_Xdata(n):
    X_train = np.random.randn(n, 2)
    return X_train

def generate_Ydata(n):
    Y_train = np.sign(np.random.rand(n))
    return Y_train

def evaluate_knn(n, algorithms):
    for i in n:
        print(f"\nDataset size: n = {i}")
        for algo in algorithms:
            clf = KNeighborsClassifier(n_neighbors=5, algorithm=algo)
            start = time.time()
            clf.fit(generate_Xdata(i), generate_Ydata(i))
            time_taken = time.time() - start
            print(f"Algorithm: {algo}, Time taken: {time_taken:.10f} seconds")

n_values = [1000, 5000, 10000, 100000]
algorithms = ['ball_tree', 'kd_tree', 'brute']
evaluate_knn(n_values, algorithms)



