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

def evaluate_knn_times(n, algorithms):
    for i in n:
        print(f"\nDataset size: n = {i}")
        for algo in algorithms:
            clf = KNeighborsClassifier(n_neighbors=5, algorithm=algo)
            start = time.time()
            clf.fit(generate_Xdata(i), generate_Ydata(i))
            time_taken = time.time() - start
            print(f"Algorithm: {algo}, Time taken: {time_taken:.10f} seconds")


def inference_test(n_test, n_values_test, algorithms):
    X_test = np.random.randn(n_test, 2)
    inference_times = {algo: [] for algo in algorithms}
    for n in n_values_test:
        X_train = generate_Xdata(n)
        Y_train = generate_Ydata(n)
        for algo in algorithms:
            clf = KNeighborsClassifier(n_neighbors=5, algorithm=algo)
            clf.fit(X_train, Y_train)
            start = time.time()
            clf.predict(X_test)
            time_taken = time.time() - start
            inference_times[algo].append(time_taken)
    return inference_times 

n_values = [1000, 5000, 10000, 100000]
n_values_test = [1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000]
algorithms = ['ball_tree', 'kd_tree', 'brute']

#part a
evaluate_knn_times(n_values, algorithms)

#part b
test_times = inference_test(5000, n_values_test, algorithms)
plt.figure(figsize=(10, 6))
for algo in algorithms:
    plt.plot(n_values_test, test_times[algo], label=f'knn-{algo}', marker='o')

plt.xlabel("Training Set Size (n)")
plt.ylabel("Inference Time (seconds)")
plt.title("kNN Inference Time vs. Training Set Size")
plt.legend()
plt.xscale("log")
plt.yscale("log")
plt.grid()
plt.show()






