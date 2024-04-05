import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt



def evaluation_metric(true_labels, predicted_labels):
    i = 0
    count = 0
    n = len(true_labels)
    while (i < n):
        if (true_labels[i] == predicted_labels[i]):
            count += 1
        i += 1
    return count / n

# Load the SPAM file and sepate data into training and validation batches
spam = np.load('spam-data.npz')
data = spam['training_data']
labels  = spam['training_labels']



size = len(data)
indices = np.arange(size)
np.random.shuffle(indices)
partitions = np.array_split(indices, 5)

c_values = [1e-09, 1e-07, 1e-05, 1e-03, 1e-01, 1,10 , 100]

for c in c_values:

    scores = 0
    svm_model = SVC(kernel='linear', C=c)
    for part in partitions:
        #validation batch
        validation_data_spam = data[part]
        validation_labes_spam = labels[part]
        #training batch
        mask = np.ones(size, dtype=bool)  # Initialize mask with all True
        mask[part] = False       # Set indices to remove to False
        # Use the mask to delete elements
        training_data_spam = data[mask]
        training_labels_spam = labels[mask]


        
        svm_model.fit(training_data_spam, training_labels_spam)
        
        predicted_labels = svm_model.predict(validation_data_spam)
        score = evaluation_metric(validation_labes_spam, predicted_labels)
        scores += score

    print("C value: " + str(c) + ", score: " + str(scores/5) + "\n")






