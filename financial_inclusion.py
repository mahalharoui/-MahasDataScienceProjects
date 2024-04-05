#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 16:45:52 2024

@author: mahalharoui
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import pickle


# Load datasets
data = pd.read_csv('/Users/mahalharoui/Documents/data science files/Financial_inclusion_dataset.csv')


# Identifying missing and corrupted values in the dataset
missing_values = data.isnull().sum()


# Checking for duplicate rows in the dataset
duplicate_rows = data.duplicated().sum()



# Iterate through each column in the dataset
for column in data.columns:
   # Check if the column is of object type
   if data[column].dtype == 'object':
       # Replace missing values with the mode
       data[column] = data[column].fillna(data[column].mode()[0])
   else:
       # For numerical columns, replace missing values with the mean
       data[column] = data[column].fillna(data[column].mean())

# Remove duplicates
data = data.drop_duplicates()



# Identifying categorical columns
categorical_columns = data.select_dtypes(include=['object']).columns


# LabelEncoding the categorical columns
# Dictionary to store original value mappings for each categorical column
original_value_mappings = {}


# LabelEncoding the categorical columns
for column in categorical_columns:
   le = LabelEncoder()
   data[column] = le.fit_transform(data[column])


   # Store the mapping of encoded labels back to original values
   original_value_mappings[column] = {index: label for index, label in enumerate(le.classes_)}

# Select target and features
X = data.drop(['bank_account', 'country', 'year','uniqueid'], axis=1)
y = data['bank_account']


# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Training the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print('Model trained')


# Predicting on the test set
y_pred = model.predict(X_test)


# Evaluating the model
accuracy = accuracy_score(y_test, y_pred)
classification_report = classification_report(y_test, y_pred)


print(f"{accuracy}\n {classification_report}")


# Save the model to a file
model_filename = 'financial_inclusion.pkl'
with open(model_filename, 'wb') as file:
   pickle.dump(model, file)