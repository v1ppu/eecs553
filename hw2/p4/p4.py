import numpy as np
import matplotlib.pyplot as plt

x = np.load("fashion_mnist_images.npy")
y = np.load("fashion_mnist_labels.npy").flatten()

lamda = 1
epsilon = 1e-6
max_iters = 1000
n = 5000

#train and test split
x_train, y_train = x[:, :n], y[:n]
y_train = y[:n]
x_test = x[:, n:]
y_test = y[n:]

theta = np.zeros((x_train.shape[0], 1))
m = x_train.shape[1]

#helper functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def J(theta):
    z = y_train * (theta.T @ x_train)
    loss = np.log(1 + np.exp(-z))
    return np.sum(loss)/n + lamda * np.linalg.norm(theta)**2

#newtons method
for iter in range(max_iters):
    y_pred = sigmoid(y_train.reshape(1,-1) * (theta.T @ x_train)).flatten()

    #gradient calculation
    grad = -(x_train @ ((1 - y_pred) * y_train).reshape(-1, 1)) / m + 2 * lamda * theta
    
    #hessian
    S = np.diag((y_pred * (1 - y_pred)).flatten())
    H = (x_train @ S @ x_train.T) / m + 2 * lamda * np.eye(x_train.shape[0])

    #newton method update
    theta -= np.linalg.inv(H) @ grad
    if(iter > 0 and abs(J_prev - J(theta)) / J_prev <= epsilon):
        break
    J_prev = J(theta)


#part a
y_test_pred = sigmoid(theta.T @ x_test).flatten()
y_test_pred_labels = (y_test_pred >= 0.5) * 2 - 1
test_error = np.mean(y_test_pred_labels != y_test)

print(f"Test error: {test_error}")
print(f"Iterations: {iter + 1}")
print(f"Final objective value: {J(theta)}")


#part b

#confidence scores
confidence = np.abs(0.5 - y_test_pred - 0.5) + 0.5

#find misclassed
misclassified = y_test_pred_labels != y_test
misclassified_indices = np.where(misclassified)[0]
misclassified_confidence = confidence[misclassified_indices]

#sort
sorted_indices = np.argsort(misclassified_confidence)
lowest_confidence_indices = misclassified_indices[sorted_indices[:20]]

# figure
plt.figure(figsize=(10, 10))
for i, idx in enumerate(lowest_confidence_indices):
    plt.subplot(4,5, i+1)

    img_size = int(np.sqrt(x_test.shape[0]))
    img = x_test[:, idx].reshape(img_size, img_size)
    plt.imshow(img, cmap='gray')
    confidence_percentage = confidence[idx] * 100
    pred_label = "Coat" if y_test_pred_labels[idx] == 1 else "Dress"
    true_label = "Coat" if y_test[idx] == 1 else "Dress"
    plt.title(f"Pred: {pred_label} ({confidence_percentage:.1f}%)\nTrue: {true_label}", fontsize=8)
    plt.axis('off')

plt.tight_layout()
plt.show()

