import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

mnist = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())

def format_data(data):
    flattened_mnist = mnist.data.numpy().astype(np.float32).reshape(-1, 784) # 784 x 1 vector
    mean = np.mean(flattened_mnist, axis=0)
    std = np.std(flattened_mnist, axis=0)  + 1e-8

    normalized_mnist = (flattened_mnist - mean) / std
    bias = np.ones((normalized_mnist.shape[0], 1), dtype=np.float32)
    return np.hstack((normalized_mnist, bias))



# task a: binary classification

# task b : original multiclass problem

#model: shallow neural network with 1 hidden layer of 
# form f(x; V,W) = V * ReLU(Wx)
# W is the input layer with h hidden neurons, V is output layers
# K = 1 for Task a, K = 10 for Task b

#initialize W anv V

def init_matrix(h, d, K):
    # W \in R(h x d), V \in R(K x h)
    W_0 = np.random.randn(h, d) * np.sqrt(2.0 / d)
    V_0 = np.random.randn(K, h) * np.sqrt(2.0 / h)
    return W_0, V_0




