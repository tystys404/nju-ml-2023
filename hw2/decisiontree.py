from sklearn.tree import DecisionTreeClassifier, export_graphviz
import numpy as np
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# Define the training dataset
X = np.array([[1, 1, 0], [1, 1, 1], [0, 0, 1], [0, 1, 0], [0, 0, 0], [1, 0, 1], [1, 1, 0], [0, 1, 1]])
y = np.array([1, 1, 0, 0, 0, 0, 1, 1])

# Create decision tree classifier with information gain criterion
clf = DecisionTreeClassifier(criterion='entropy')
clf.fit(X, y)

# Predict the class labels of the training dataset
y_pred = clf.predict(X)

# Calculate the accuracy of the model
accuracy = np.mean(y_pred == y) * 100
print("Accuracy:", accuracy)

plt.figure(figsize=(10,5))
plot_tree(clf,filled=True,feature_names=["X1","X2","X3"],class_names=["class1","class2"])
plt.show()
