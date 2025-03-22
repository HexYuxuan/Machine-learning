import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
# Load the dataset
data = np.loadtxt('seeds_dataset.txt')

# Convert the data to a DataFrame
df = pd.DataFrame(data, columns=['Area', 'Perimeter', 'Compactness', 'Length of Kernel', 'Width of Kernel', 'Asymmetry Coefficient', 'Length of Kernel Groove', 'Class'])

# Split the dataset into features (X) and target variable (y)
X = df.drop('Class', axis=1)
y = df['Class']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# LDA model
clf = LinearDiscriminantAnalysis()
model=clf.fit(X_train,y_train)

pred_class_train = model.predict(X_train)
pred_class_test = model.predict(X_test)

# Training Data Probability Prediction
pred_prob_train = model.predict_proba(X_train)

# Test Data Probability Prediction
pred_prob_test = model.predict_proba(X_test)

# Accuracy - Training Data
print(model.score(X_train, y_train))

# AUC and ROC for the training data

# 计算AUC
# 计算多类别AUC
auc_test = metrics.roc_auc_score(y_test, pred_prob_test, multi_class='ovr')
auc_train = metrics.roc_auc_score(y_train, pred_prob_train, multi_class='ovr')
print('AUC for the Test Data: %.3f' % auc_test)
print('AUC for the Train Data: %.3f' % auc_train)


# 计算ROC曲线
fpr_test = dict()
tpr_test = dict()
thresholds_test = dict()
for i, class_label in enumerate(np.unique(y_test)):
    fpr_test[class_label], tpr_test[class_label], thresholds_test[class_label] = metrics.roc_curve(
        np.where(y_test == class_label, 1, 0), pred_prob_test[:, i]
    )

# 绘制ROC曲线
plt.figure()
for class_label in np.unique(y_test):
    plt.plot(fpr_test[class_label], tpr_test[class_label], marker='.', label='Class %d' % class_label)
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='best')
plt.show()

f,a =  plt.subplots(1,2,sharex=True,sharey=True,squeeze=False)

#Plotting confusion matrix for the different models for the Training Data

plot_0 = sns.heatmap((metrics.confusion_matrix(y_train,pred_class_train)),annot=True,fmt='.5g',cmap='Greys',ax=a[0][0])
a[0][0].set_title('Training Data')

plot_1 = sns.heatmap((metrics.confusion_matrix(y_test,pred_class_test)),annot=True,fmt='.5g',cmap='Greys',ax=a[0][1])
a[0][1].set_title('Test Data')
plt.show()