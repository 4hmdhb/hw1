import numpy as np
from sklearn.svm import SVC





# separate MNIST data into validation and training batches, also flatten image
mnist = np.load('mnist-data.npz')
data = mnist['training_data']
data = data.reshape(60000,-1)
labels = mnist['training_labels']
indices = np.random.choice(range(0, 60000), 10000, replace=False)
validation_data_mnist = data[indices]
validation_labes_mnist = labels[indices]

#training batch
mask = np.ones(60000, dtype=bool)  # Initialize mask with all True
mask[indices] = False       # Set indices to remove to False
# Use the mask to delete elements
training_data_mnist = data[mask]
training_labels_mnist = labels[mask]



def evaluation_metric(true_labels, predicted_labels):
    i = 0
    count = 0
    n = len(true_labels)
    while (i < n):
        if (true_labels[i] == predicted_labels[i]):
            count += 1
        i += 1
    return count / n




c_values = [1e-09, 1e-07, 1e-05, 1e-03, 1e-01, 1, 100, 10000]
select_data_points = np.random.choice(range(0, 50000), 10000, replace=False)

selected_training_data = training_data_mnist[select_data_points]
selected_training_labels = training_labels_mnist[select_data_points]
for c in c_values:

    svm_model = SVC(kernel='linear', C=c)
    svm_model.fit(selected_training_data,selected_training_labels)
    
    predicted_labels_mnist = svm_model.predict(validation_data_mnist)
    score = evaluation_metric(validation_labes_mnist, predicted_labels_mnist)
    print("C value: " + str(c) + ", score: " + str(score) + "\n")

