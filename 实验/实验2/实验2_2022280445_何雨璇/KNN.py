from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier as KNN, KNeighborsClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from  IPython import display
from matplotlib import pyplot as plt
import numpy as np
import pathlib
import shutil
import tempfile
from tqdm import tqdm
import pandas as pd
# Load the dataset
data = np.loadtxt('seeds_dataset.txt')

# Convert to DataFrame
df = pd.DataFrame(data, columns=['Area', 'Perimeter', 'Compactness', 'Length of Kernel', 'Width of Kernel',
                                 'Asymmetry Coefficient', 'Length of Kernel Groove', 'Class'])

# Separate features (X) and target variable (y)
X = df.drop('Class', axis=1).values
y = df['Class'].values

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Define the list of K values
k_values = [1, 5, 10, 20, 50, 70, 79]
accuracy = []

# Initialize lists to store the training and testing errors
train_errors = []
test_errors = []

# Iterate over the given K values
for k in k_values:
    # Create a KNN classifier object
    knn = KNeighborsClassifier(n_neighbors=k)

    # Train the classifier
    knn.fit(X_train, y_train)

    # Get accuracy
    y_pred = knn.predict(X_test)
    accuracy.append(metrics.accuracy_score(y_test, y_pred))

    # Calculate training error
    train_error = 1 - knn.score(X_train, y_train)
    train_errors.append(train_error)

    # Calculate testing error
    test_error = 1 - knn.score(X_test, y_test)
    test_errors.append(test_error)

# Show the accuracy
print(accuracy)


# Plot the training and testing errors
plt.plot(k_values, train_errors, label='Training Error')
plt.plot(k_values, test_errors, label='Testing Error')
plt.xlabel('K')
plt.ylabel('Error')
plt.title('Training and Testing Errors vs. K')
plt.legend()
plt.show()

#cross_validate
scores = []
for k in k_values:
    knn = KNN(n_neighbors=k)
    score = cross_val_score(knn, X_train, y_train, cv=5)
    scores.append(1 - score.mean())
plt.figure()
plt.plot(k_values, scores, 'ko-')
min_k = k_values[np.argmin(scores)]
plt.plot([min_k, min_k], [0, 1.0], 'b-')
plt.xlabel('k')
plt.ylabel('misclassification rate')
plt.title('5-fold cross validation, n-train = 200')
plt.show()