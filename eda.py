import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_boston

# Load the dataset
boston = load_boston()
data = pd.DataFrame(boston.data, columns=boston.feature_names)
data['PRICE'] = boston.target

# Display the first few rows of the dataset
print("First few rows of the dataset:")
print(data.head())

# Display summary statistics
print("\nSummary statistics:")
print(data.describe())

# Check for missing values
print("\nMissing values in the dataset:")
print(data.isnull().sum())

# Correlation matrix
corr_matrix = data.corr()

# Plotting
plt.figure(figsize=(12, 8))

# Histogram of each feature
data.hist(bins=20, figsize=(20, 15), color='b', edgecolor='k')
plt.suptitle('Histogram of Each Feature')
plt.show()

# Scatter plot of RM vs PRICE
plt.figure(figsize=(8, 6))
sns.scatterplot(x=data['RM'], y=data['PRICE'])
plt.title('Scatter Plot: RM vs PRICE')
plt.xlabel('Average number of rooms per dwelling (RM)')
plt.ylabel('House Price')
plt.show()

# Heatmap of the correlation matrix
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix Heatmap')
plt.show()

# Pairplot of features vs PRICE
plt.figure(figsize=(12, 8))
sns.pairplot(data[['RM', 'LSTAT', 'PTRATIO', 'PRICE']])
plt.suptitle('Pairplot of Selected Features vs PRICE')
plt.show()

# Boxplot to identify outliers in PRICE
plt.figure(figsize=(8, 6))
sns.boxplot(x=data['PRICE'])
plt.title('Boxplot of House Prices')
plt.show()
