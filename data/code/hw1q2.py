import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC



toy = np.load('toy-data.npz')
data = toy['training_data']

labels = toy['training_labels']


svm = SVC(kernel='linear')
svm.fit(data, labels)

w = svm.coef_[0]
b = svm.intercept_[0]



plt.scatter(data[:, 0], data[:, 1], c=labels)
# Plot the decision boundary
x = np.linspace(-5, 5, 100)
y = -(w[0] * x + b) / w[1]


plt.plot(x, y, 'k')
# Plot the margins


margin = 1 / np.linalg.norm(w)
unit_w = w / np.linalg.norm(w)
decision_boundary_points = np.array(list(zip(x, y)))

points_of_line_above = decision_boundary_points + unit_w * margin
points_of_line_below = decision_boundary_points - unit_w * margin

plt.plot(points_of_line_above[:, 0], points_of_line_above[:, 1])

plt.plot(points_of_line_below[:, 0], points_of_line_below[:, 1])


plt.plot(x, y, 'k')

plt.show()
         
