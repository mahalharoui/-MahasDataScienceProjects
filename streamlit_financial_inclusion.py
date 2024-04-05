import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder

# Load your trained model
model = pickle.load(open('/Users/mahalharoui/.spyder-py3/financial_inclusion.pkl', 'rb'))

# Load datasets
data_path = '/Users/mahalharoui/Documents/data science files/Financial_inclusion_dataset.csv'
original_data = pd.read_csv(data_path)

# Identifying categorical columns
categorical_columns = original_data.select_dtypes(include=['object']).columns

# Dictionaries to store the label encoders for each categorical column
label_encoders = {}

# Create a copy of the data for manipulation
data = original_data.copy()

# LabelEncoding the categorical columns
for column in categorical_columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# Define the layout of the application
st.title('Bank Account Prediction')
st.write('Enter the details to predict financial inclusion')

# Input fields for the selected features
location = st.selectbox('location_type', original_data['location_type'].unique())
cellphone = st.selectbox('cellphone_access', original_data['cellphone_access'].unique())
h_size = st.number_input('household_size', min_value=0.0, format='%f')
age = st.number_input('age_of_respondent', min_value=0, format='%d')
gender = st.selectbox('gender_of_respondent', original_data['gender_of_respondent'].unique())
relationship_head = st.selectbox('relationship_with_head', original_data['relationship_with_head'].unique())
status = st.selectbox('marital_status', original_data['marital_status'].unique())
education = st.selectbox('education_level', original_data['education_level'].unique())
job = st.selectbox('job_type', original_data['job_type'].unique())

# Button to make predictions
if st.button('Predict'):
    # Create a DataFrame with the input features
    input_data = pd.DataFrame([[location, cellphone, h_size, age, gender, relationship_head, status, education, job]],
                              columns=['location_type', 'cellphone_access', 'household_size', 'age_of_respondent', 'gender_of_respondent', 'relationship_with_head', 'marital_status', 'education_level', 'job_type'])

    # Map the selected values to their encoded form
    for column in categorical_columns:
        if column in input_data:
            input_data[column] = label_encoders[column].transform(input_data[column])

    # Make prediction
    prediction = model.predict(input_data)

    # Interpret and Display the prediction
    prediction_result = "Yes" if prediction[0] == 1 else "No"
    st.write('Bank Account Prediction: ', prediction_result)
    
