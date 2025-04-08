import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

def format_data(data):
    flattened_mnist = data.astype(np.float32).reshape(-1, 784)
    mean = np.mean(flattened_mnist, axis=0)
    std = np.std(flattened_mnist, axis=0)  + 1e-8

    normalized_mnist = (flattened_mnist - mean) / std
    bias = np.ones((normalized_mnist.shape[0], 1), dtype=np.float32)
    return np.hstack((normalized_mnist, bias))

def init_matrix(h, d, K):
    # W \in R(h x d), V \in R(K x h)
    W_0 = np.random.randn(h, d) * np.sqrt(2.0 / d)
    V_0 = np.random.randn(K, h) * np.sqrt(2.0 / h)
    return W_0, V_0

def one_hot_encode(y, num_classes=10):
    return np.eye(num_classes)[y.astype(int)]

def compute_gradient(x, y, W, V, loss_type=''):
    # x : input data
    # y : label
    # W ; weight matrix for hidden layer (h x d)
    # V : weight matrix for output layer (K x h)
    # loss type : quadratic or logistic
    # return dV, dW

    size = x.shape[0]
    z = np.maximum(0, np.dot(x, W.T)) #ReLU(Wx)
    y_pred = np.dot(z, V.T)

    if(loss_type == 'quadratic'):
        # L = 1/2 * ||y - y_pred||^2
        dL = y_pred - y.reshape(-1,1)
    elif(loss_type == 'logistic'):
        # L = -y * log(sigma*f(x))-(1-y)log(1-sigma(f(x)))
        sigmoid = 1/(1+np.exp(-y_pred))
        dL = sigmoid - y.reshape(-1,1)

    #back prop
    dV = np.dot(dL.T, z) / size
    dZ = np.dot(dL, V)
    relu_deriv = (z > 0).astype(float)
    dZ = dZ * relu_deriv
    dW = np.dot(dZ.T, x) / size
    return dW, dV


def compute_multiclass_gradient(x, y, W, V):
    # l(y,f(x)) = - log(softmax(f(x))_y)

    # x : input data
    # y : label
    # W ; weight matrix for hidden layer (h x d)
    # V : weight matrix for output layer (K x h)
    # loss type : quadratic or logistic
    # return dV, dW

    size = x.shape[0]
    z = np.maximum(0, np.dot(x, W.T)) #ReLU(Wx)
    logits = np.dot(z, V.T)
    
    exps = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probs = exps / np.sum(exps, axis=1, keepdims=True)
    dL = (probs - y)/size

    #back prop
    dV = np.dot(dL.T, z) / size
    dZ = np.dot(dL, V)
    relu_deriv = (z > 0).astype(float)
    dZ = dZ * relu_deriv
    dW = np.dot(dZ.T, x) / size
    return dW, dV


# task a: binary classification
def train(x_train, y_train, x_test, y_test, h, loss_type='', 
    learning_rate = 0.001, batch_size = 16, epochs = 8, report_freq = 100):
    
    d = x_train.shape[1]
    W, V = init_matrix(h, d, 1)
    train_accuracy_list = []
    test_accuracy_list = []
    inters = []

    n_iter = int((x_train.shape[0] / batch_size) * epochs)

    y_train_bin = (y_train > 4).astype(int)
    y_test_bin = (y_test > 4).astype(int)

    for iter in range(n_iter):
        #sample mini-batch
        idx = np.random.choice(x_train.shape[0], batch_size, replace=False)
        x_batch = x_train[idx]
        y_batch = y_train_bin[idx]

        #gradients
        dW, dV = compute_gradient(x_batch, y_batch, W, V, loss_type) 

        #update
        W -= learning_rate * dW
        V -= learning_rate * dV

        if iter % report_freq == 0 or n_iter == -1 :
            train_acc = eval_acc(x_train, y_train_bin, W, V)
            test_acc = eval_acc(x_test, y_test_bin, W, V)
    
            train_accuracy_list.append(train_acc)
            test_accuracy_list.append(test_acc)
            inters.append(iter)

            print(f"Iteration {iter}: Train Accuracy: = {train_acc:.4f}, Test Accuracy: {test_acc:.4f}")

    return W, V, train_accuracy_list, test_accuracy_list, inters

def train_multiclass(x_train, y_train, x_test, y_test, h,
    learning_rate=0.001, batch_size=16, epochs=8, report_freq=100):

    d = x_train.shape[1]
    W, V = init_matrix(h, d, 10)
    train_accuracy_list = []
    test_accuracy_list = []
    inters = []

    n_iter = int((x_train.shape[0] / batch_size) * epochs)

    y_train_multi = one_hot_encode(y_train, 10)
    y_test_multi = one_hot_encode(y_test, 10)

    for iter in range(n_iter):
        #sample mini-batch
        idx = np.random.choice(x_train.shape[0], batch_size, replace=False)
        x_batch = x_train[idx]
        y_batch = y_train_multi[idx]

        #gradients
        dW, dV = compute_multiclass_gradient(x_batch, y_batch, W, V) 

        #update
        W -= learning_rate * dW
        V -= learning_rate * dV

        if iter % report_freq == 0 or n_iter == -1 :
            train_acc = eval_multiclass_acc(x_train, y_train, W, V)
            test_acc = eval_multiclass_acc(x_test, y_test, W, V)
    
            train_accuracy_list.append(train_acc)
            test_accuracy_list.append(test_acc)
            inters.append(iter)

            print(f"Iteration {iter}: Train Accuracy: = {train_acc:.4f}, Test Accuracy: {test_acc:.4f}")

    return W, V, train_accuracy_list, test_accuracy_list, inters

def eval_acc(x, y, W, V):
    # x : input data
    # y : label
    # W ; weight matrix for hidden layer (h x d)
    # V : weight matrix for output layer (K x h)

    z = np.maximum(0, np.dot(x, W.T)) #ReLU(Wx)
    y_pred = np.dot(z, V.T)
    predictions = (y_pred > 0.5).astype(int).flatten()
    accuracy = np.mean(predictions == y)
    return accuracy

def eval_multiclass_acc(x,y,W,V):
    # x : input data
    # y : label
    # W ; weight matrix for hidden layer (h x d)
    # V : weight matrix for output layer (K x h)

    z = np.maximum(0, np.dot(x, W.T)) #ReLU(Wx)
    y_pred = np.dot(z, V.T)
    predictions = np.argmax(y_pred, axis=1)
    accuracy = np.mean(predictions == y)
    return accuracy


def plot_acc(train_acc, test_acc, inters, h):
    #plot train/test acc
    plt.figure(figsize=(10, 6))
    plt.plot(inters, train_acc, 'b-', label='Training Accuracy')
    plt.plot(inters, test_acc, 'r-', label='Test Accuracy')
    plt.title(f'Accuracy vs. Iterations (h={h})')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()

mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())

x_train = format_data(mnist_train.data.numpy())
y_train = mnist_train.targets.numpy()
x_test = format_data(mnist_test.data.numpy())
y_test = mnist_test.targets.numpy()


print(f"Training using quadratic loss function")
for h in [5, 40, 200]:
    print(f"Training with hidden layer size: {h}")
    W, V, train_acc, test_acc, inters = train(x_train,y_train,x_test,y_test,h=h,loss_type='quadratic',report_freq=100)

    plot_acc(train_acc, test_acc, inters, h)
    print(f"Final test acc with h = {h}: {test_acc[-1]:.4f}")


print(f"Training using logistic loss function")
for h in [5, 40, 200]:
    print(f"Training with hidden layer size: {h}")
    W, V, train_acc, test_acc, inters = train(x_train,y_train,x_test,y_test,h=h,loss_type='logistic',report_freq=100)

    plot_acc(train_acc, test_acc, inters, h)
    print(f"Final test acc with h = {h}: {test_acc[-1]:.4f}")

print(f"Multiclass Classification")
for h in [5, 40, 200]:
    print(f"Training with hidden layer size: {h}")
    W, V, train_acc, test_acc, inters = train_multiclass(x_train,y_train,x_test,y_test,h=h,report_freq=100)

    plot_acc(train_acc, test_acc, inters, h)
    print(f"Final test acc with h = {h}: {test_acc[-1]:.4f}")