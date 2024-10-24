#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Import necessary libraries
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.stat import Correlation

# Step 1: Load the dataset using Pandas with the correct delimiter
file_path = 'cardio_train.csv'  # Assuming the file is in the same directory as the notebook
df = pd.read_csv(file_path, delimiter=';')  # Specify the delimiter as semicolon

# Display the first few rows of the dataset
print("Pandas DataFrame head:")
print(df.head())

# Step 2: Initialize Spark session
spark = SparkSession.builder \
    .appName("CardiovascularDataAnalysis") \
    .master("local[*]") \
    .getOrCreate()

# Step 3: Convert Pandas DataFrame to Spark DataFrame
spark_df = spark.createDataFrame(df)

# Show the Spark DataFrame
print("Spark DataFrame schema:")
spark_df.printSchema()

# Step 4: Data Preprocessing
# Dropping rows with missing values
cleaned_data = spark_df.dropna()

# Selecting relevant features for analysis
feature_columns = ['age', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'smoke', 'alco', 'active']
assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
assembled_data = assembler.transform(cleaned_data)

# Step 5: Standardize the feature vectors
scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures", withStd=True, withMean=False)
scaler_model = scaler.fit(assembled_data)
scaled_data = scaler_model.transform(assembled_data)

# Step 6: Correlation Analysis
# Compute correlation matrix for the features
correlation_matrix = Correlation.corr(scaled_data, "scaledFeatures").head()
print("Correlation matrix:\n", correlation_matrix[0].toArray())

# Step 7: Show the results
print("Scaled Features:")
scaled_data.select("scaledFeatures").show(5)

# Stop the Spark session
spark.stop()


# In[2]:


# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = 'cardio_train.csv'
df = pd.read_csv(file_path, delimiter=';')  # Use the correct delimiter

# Display the first few rows of the dataset
print(df.head())

# Step 1: Plot histograms for numerical predictors
numerical_features = ['age', 'height', 'weight', 'ap_hi', 'ap_lo']
df[numerical_features].hist(bins=30, figsize=(12, 10), grid=False)
plt.suptitle('Histograms of Numerical Predictors')
plt.show()

# Step 2: Plot bar plots for categorical predictors
categorical_features = ['gender', 'cholesterol', 'gluc', 'smoke', 'alco', 'active']
plt.figure(figsize=(12, 8))
for i, feature in enumerate(categorical_features, 1):
    plt.subplot(3, 2, i)
    sns.countplot(x=feature, data=df)
    plt.title(f'Frequency Distribution of {feature}')
plt.tight_layout()
plt.show()

# Step 3: Create boxplots to check for outliers among numerical features
plt.figure(figsize=(12, 8))
for i, feature in enumerate(numerical_features, 1):
    plt.subplot(3, 2, i)
    sns.boxplot(x=df[feature])
    plt.title(f'Boxplot of {feature}')
plt.tight_layout()
plt.show()

# Step 4: Generate a correlation matrix to examine relationships
correlation_matrix = df[numerical_features + ['cardio']].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title('Correlation Matrix')
plt.show()


# In[ ]:




