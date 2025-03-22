import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Obtaining the Boston house price dataset from the original data source
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]
feature = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']

# Check for missing values
if raw_df.isnull().values.any():
    print("Missing values in the dataset")
else:
    print("No missing values in the data set")

# Divide the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.1, random_state=42)
# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize the ridge regression model
ridge = Ridge()
# Determining the optimal alpha value using 10-fold cross-validation
params = {'alpha': [0.01, 0.1, 1, 10, 100]}
grid_search = GridSearchCV(ridge, params, cv=10, scoring='neg_mean_squared_error')
grid_search.fit(X_train_scaled, y_train)
# Output the optimal alpha value
print("the optimal alpha value：", grid_search.best_params_['alpha'])

# Prediction using a model with optimal alpha value
y_pred = grid_search.predict(X_test_scaled)
# Calculate the MSE on the test set
test_mse = mean_squared_error(y_test, y_pred)
print("the MSE on the test set：", test_mse)

# Exploratory data analysis
# Distribution of target variables
plt.figure(figsize=(10, 6))
sns.histplot(target, kde=True)
plt.title('House Prices Distribution')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.show()

# correlation matrix
plt.figure(figsize=(10, 8))
X_df = pd.DataFrame(data, columns=[feature[i] for i in range(data.shape[1])])
corr_matrix = X_df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Scatterplot of raw vs. predicted house prices
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', label='Predicted Prices')
plt.scatter(y_test, y_test, color='red', label='Actual Prices')
plt.plot(y_test, y_test, color='green', label='Right Line')

# Plotting the horizontal line of the actual average price
mean_price = np.mean(y_test)
plt.hlines(y=mean_price, xmin=0, xmax=np.max(y_test), colors='black', linestyle='--')

# Plotting the horizontal line of the predicted average price
mean_pred_price = np.mean(y_pred)
plt.hlines(y=mean_pred_price, xmin=0, xmax=np.max(y_pred), colors='black', linestyle='--')

plt.title('Actual Prices vs Predicted Prices')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.legend()
plt.show()