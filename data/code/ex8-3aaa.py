import numpy as np
from sklearn.model_selection import train_test_split
from scipy.linalg import solve

# Define the function to fit Gaussian to digit classes
def fit_gaussian_to_digit_class(data, labels):
    means = []
    covariances = []
    for digit in range(10): # There are 10 classes for digits 0-9
        # Extract the data for the current digit
        digit_data = data[labels == digit]
        # Compute the mean and covariance matrix for the current digit
        mean = np.mean(digit_data, axis=0)
        covariance = np.cov(digit_data, rowvar=False)
        # Add a small value to the diagonal to prevent singular matrix error
        covariance += np.eye(covariance.shape[0]) * 1e-6
        means.append(mean)
        covariances.append(covariance)
    return means, covariances

# Load the dataset from the uploaded file
data = np.load('mnist-data-hw3.npz')
training_data = data['training_data']
training_labels = data['training_labels']
test_data = data['test_data']

# Normalize the training and test data
training_data_reshaped = training_data.reshape(training_data.shape[0], -1)  # Flatten the images
norms = np.linalg.norm(training_data_reshaped, axis=1, keepdims=True)
norms[norms == 0] = 1
training_data_normalized = training_data_reshaped / norms

test_data_reshaped = test_data.reshape(test_data.shape[0], -1)  # Flatten the images
test_norms = np.linalg.norm(test_data_reshaped, axis=1, keepdims=True)
test_norms[test_norms == 0] = 1
test_data_normalized = test_data_reshaped / test_norms

# Split the training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    training_data_normalized, training_labels, test_size=10000, random_state=42)

# Fit Gaussian to training data (not validation data) to get means and pooled covariance matrix
means, covariances = fit_gaussian_to_digit_class(X_train, y_train)

# Calculate the pooled covariance matrix (weighted average of covariance matrices)
n_features = training_data_normalized.shape[1]
pooled_covariance = np.zeros((n_features, n_features))
for i in range(10):
    pooled_covariance += covariances[i] * (y_train == i).sum()
pooled_covariance /= X_train.shape[0]

# Function to compute the LDA score for each class
def lda_score(x, mean, pooled_covariance_inv):
    return np.dot(x, np.dot(pooled_covariance_inv, mean)) - 0.5 * np.dot(mean, np.dot(pooled_covariance_inv, mean))

# Predict using LDA
pooled_covariance_inv = np.linalg.inv(pooled_covariance)
predictions = []
for sample in X_val:
    scores = [lda_score(sample, mean, pooled_covariance_inv) for mean in means]
    predictions.append(np.argmax(scores))

# Calculate the error rate
error_rate = np.mean(np.array(predictions) != y_val)
error_rate
print(error_rate)