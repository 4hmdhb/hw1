import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import csv


# Load the SPAM file and sepate data into training and validation batches
spam = np.load('spam-data.npz')
data2 = spam['training_data']
labels2 = spam['training_labels']
test_data = spam['test_data']

size = len(data2)
sample_size = int(size * 10 / 100)
indices2 = np.random.choice(range(0, size), sample_size, replace=False)

#validation batch
validation_data_spam = data2[indices2]
validation_labes_spam = labels2[indices2]

#training batch
mask = np.ones(size, dtype=bool)  # Initialize mask with all True
mask[indices2] = False       # Set indices to remove to False

# Use the mask to delete elements
training_data_spam = data2[mask]
training_labels_spam = labels2[mask]



#train SVM using SKLEARN on SPAM dataset
svm_model_spam = SVC(kernel='rbf', C=10000, gamma=0.0001)
svm_model_spam.fit(data2, labels2)



'''

def evaluation_metric(true_labels, predicted_labels):
    i = 0
    count = 0
    n = len(true_labels)
    while (i < n):
        if (true_labels[i] == predicted_labels[i]):
            count += 1
        i += 1
    return count / n

predicted_labels_spam = svm_model_spam.predict(validation_data_spam)
score = evaluation_metric(validation_labes_spam, predicted_labels_spam)
print(score)

'''

predicted_labels_spam = svm_model_spam.predict(test_data)
with open('example_spam.csv', mode='w', newline='\n') as file:
    writer = csv.writer(file)
    writer.writerow(["Id", "Category"])
    id = 1
    for categ in predicted_labels_spam:
        writer.writerow([id, categ])
        id += 1


