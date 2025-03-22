import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
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

# GNB
gnb = GaussianNB()
y_pred = gnb.fit(X_train, y_train).predict(X_test)

print("Number of mislabeled points out of a total {} points :{}".format(X_test.shape[0], (y_test != y_pred).sum()))
print(gnb.score(X,y))
print("Accuracy of GNB:",accuracy_score(y_test,y_pred))

# MNB
clf = MultinomialNB()
y_predm = clf.fit(X_train, y_train).predict(X_test)

print("Number of mislabeled points out of a total {} points :{}".format(X_test.shape[0], (y_test != y_predm).sum()))
print(clf.score(X,y))
print("Accuracy of MNB:",accuracy_score(y_test,y_predm))

# BNB
clf2 = BernoulliNB()
y_predb = clf2.fit(X_train, y_train).predict(X_test)

print("Number of mislabeled points out of a total {} points :{}".format(X_test.shape[0], (y_test != y_predb).sum()))
print(clf2.score(X,y))
print("Accuracy of BNB:",accuracy_score(y_test,y_predb))