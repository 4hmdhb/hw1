import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt


# Load the SPAM file and sepate data into training and validation batches
spam = np.load('spam-data.npz')
data2 = spam['training_data']
labels2 = spam['training_labels']
size = len(data2)
sample_size = int(size * 20 / 100)
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


def evaluation_metric(true_labels, predicted_labels):
    i = 0
    count = 0
    n = len(true_labels)
    while (i < n):
        if (true_labels[i] == predicted_labels[i]):
            count += 1
        i += 1
    return count / n


#train SVM using SKLEARN on SPAM dataset
svm_model_spam = SVC(kernel='linear')
training_scores_spam = []
validation_scores_spam = []
sizes2 = [100, 200, 500, 1000, 2000, size-sample_size]
for i in sizes2:
    select_data_points = np.random.choice(range(0, size - sample_size), i, replace= False)

    selected_training_data = training_data_spam[select_data_points]
    selected_training_labels = training_labels_spam[select_data_points]

    svm_model_spam.fit(selected_training_data, selected_training_labels)


    predicted_labels_spam_v = svm_model_spam.predict(validation_data_spam)
    score = evaluation_metric(validation_labes_spam, predicted_labels_spam_v)
    validation_scores_spam.append(score)

    predicted_labels_spam_t = svm_model_spam.predict(selected_training_data)
    score2 = evaluation_metric(selected_training_labels, predicted_labels_spam_t)
    training_scores_spam.append(score2)

plt.plot(sizes2, validation_scores_spam, label='validation score')
plt.plot(sizes2, training_scores_spam, label='training score')
plt.xlabel("sample size")
plt.legend()
plt.ylabel("accuracy")
plt.title('Spam')
plt.show()