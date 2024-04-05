import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import csv


def evaluation_metric(true_labels, predicted_labels):
    i = 0
    count = 0
    n = len(true_labels)
    while (i < n):
        if (true_labels[i] == predicted_labels[i]):
            count += 1
        i += 1
    return count / n


# separate MNIST data into validation and training batches, also flatten image
mnist = np.load('mnist-data.npz')
data = mnist['training_data']
data = data.reshape(60000,-1)
labels = mnist['training_labels']

test_data = mnist['test_data']
test_data = test_data.reshape(len(test_data),-1)
indices = np.random.choice(range(0, 60000), 10000, replace=False)
validation_data_mnist = data[indices]
validation_labes_mnist = labels[indices]

#training batch
mask = np.ones(60000, dtype=bool)  # Initialize mask with all True
mask[indices] = False       # Set indices to remove to False
# Use the mask to delete elements
training_data_mnist = data[mask]
training_labels_mnist = labels[mask]









#train SVM using SKLEARN on MNIST dataset
svm_model = SVC(kernel='linear', C=1e-07)
svm_model.fit(training_data_mnist[:11000], training_labels_mnist[:11000])
predicted_labels_mnist = svm_model.predict(test_data)






with open('example.csv', mode='w', newline='\n') as file:
    writer = csv.writer(file)
    writer.writerow(["Id", "Category"])
    id = 1
    for categ in predicted_labels_mnist:
        writer.writerow([id, categ])
        id += 1
