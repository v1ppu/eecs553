import numpy as np
import matplotlib.pyplot as plt

#part c
train_x = np.load("hw2p2_train_x.npy") # dimensions: n_train x d (1192 x 1000)
train_y = np.load("hw2p2_train_y.npy") # dimensions: n_train x 1 vector of labels (1192 x 1)

#n_train = 1192 samples
#n_test = 794 samples
#d = 1000 features (each feature denotes the number of times a word appeared in a document)

n_train, d = train_x.shape
alpha = 1

class_0_indices = (train_y == 0)
class_1_indices = (train_y == 1)

n_k0 = np.sum(train_x[class_0_indices]) # total words class 0
n_k1 = np.sum(train_x[class_1_indices]) # total words class 1

n_kj0 = np.sum(train_x[class_0_indices], axis=0)
n_kj1 = np.sum(train_x[class_1_indices], axis=0)

#p_kj = (n_kj + alpha) / (n_k + alpha*d) formula from derivation
p_kj0 = (n_kj0 + alpha) / (n_k0 + alpha*d)
p_kj1 = (n_kj1 + alpha) / (n_k1 + alpha*d)
log_p_kj0 = np.log(p_kj0)
log_p_kj1 = np.log(p_kj1)


# log(pi_hat_k) = log(m_k / n) formula from derivation
m_k0 = np.sum(class_0_indices)
m_k1 = np.sum(class_1_indices)
n = n_train

pi_hat_0 = m_k0 / n
pi_hat_1 = m_k1 / n
log_pi_hat_0 = np.log(pi_hat_0)
log_pi_hat_1 = np.log(pi_hat_1)

#(i) solution
np.savetxt("log_p_kj0.csv", log_p_kj0, delimiter=",")
np.savetxt("log_p_kj1.csv", log_p_kj1, delimiter=",")

#(ii) solution
print("Log priors:")
print(f"log π_0: {log_pi_hat_0}, log π_1: {log_pi_hat_1}")

# part d
test_x = np.load("hw2p2_test_x.npy")
test_y = np.load("hw2p2_test_y.npy")

#log probabilities for each class
log_prob_0 = log_pi_hat_0 + test_x @ log_p_kj0
log_prob_1 = log_pi_hat_1 + test_x @ log_p_kj1

predict = (log_prob_1 > log_prob_0).astype(int)

test_error = np.mean(predict != test_y)
print(f"Test error: {test_error}")

#part e, if always predict more majority class
majority_class = np.bincount(train_y).argmax()
baseline = np.full_like(test_y, majority_class)
baseline_error = np.mean(baseline != test_y)
print(f"Baseline error (when always predicting class {majority_class}): {baseline_error}")
