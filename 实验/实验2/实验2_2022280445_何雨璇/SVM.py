import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load the dataset
data = np.loadtxt('seeds_dataset.txt')

# Convert to DataFrame
df = pd.DataFrame(data, columns=['Area', 'Perimeter', 'Compactness', 'Length of Kernel', 'Width of Kernel', 'Asymmetry Coefficient', 'Length of Kernel Groove', 'Class'])

# Separate features (X) and target variable (y)
X = df.drop('Class', axis=1)
y = df['Class']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Create an SVM classifier object
svm = SVC()

# Train the classifier
svm.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = svm.predict(X_test)

# Calculate the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Create separate scatter plots for the testing set and the predicted set
plt.figure(figsize=(12, 6))

# Plot the positions of the different classes in the testing set
plt.subplot(1, 2, 1)
for c in np.unique(y_test):
    plt.scatter(X_test.loc[y_test == c, 'Area'], X_test.loc[y_test == c, 'Perimeter'], label=f'Class {int(c)}')
plt.xlabel('Area')
plt.ylabel('Perimeter')
plt.title('Positions of Classes in the Testing Set')
plt.legend()

# Plot the positions of the different classes in the predicted set
plt.subplot(1, 2, 2)
for c in np.unique(y_pred):
    plt.scatter(X_test.loc[y_pred == c, 'Area'], X_test.loc[y_pred == c, 'Perimeter'], label=f'Class {int(c)}')
plt.xlabel('Area')
plt.ylabel('Perimeter')
plt.title('Positions of Classes in the Predicted Set')
plt.legend()

plt.tight_layout()
plt.show()