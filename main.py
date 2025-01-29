import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load or generate the dataset (replace this with real data)
def load_data():
    # Sample dataset with hypothetical features
    data = pd.DataFrame({
        'Age': np.random.randint(2, 50, size=100),
        'Gender': np.random.choice(['Male', 'Female'], size=100),
        'Communication Skills': np.random.randint(1, 10, size=100),
        'Social Interaction': np.random.randint(1, 10, size=100),
        'Repetitive Behaviors': np.random.randint(1, 10, size=100),
        'Attention to Detail': np.random.randint(1, 10, size=100),
        'ASD Diagnosis': np.random.choice([0, 1], size=100)  # 0: No ASD, 1: ASD
    })
    return data

# Preprocess the data (encode categorical variables)
def preprocess_data(df):
    df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})  # Encode 'Gender'
    X = df.drop('ASD Diagnosis', axis=1)  # Features
    y = df['ASD Diagnosis']  # Target
    return X, y

# Train a simple RandomForest model (replace with a pre-trained model if available)
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return model, acc

# Streamlit app layout
def main():
    st.title("Autism Prediction App")

    # Load and preprocess the data
    data = load_data()
    X, y = preprocess_data(data)
    model, accuracy = train_model(X, y)

    # Show the accuracy of the model
    st.write(f"Model Accuracy: {accuracy * 100:.2f}%")

    # Sidebar for user input
    st.sidebar.header("Input Features")
    age = st.sidebar.slider('Age', 2, 50, 25)
    gender = st.sidebar.selectbox('Gender', ('Male', 'Female'))
    communication_skills = st.sidebar.slider('Communication Skills', 1, 10, 5)
    social_interaction = st.sidebar.slider('Social Interaction', 1, 10, 5)
    repetitive_behaviors = st.sidebar.slider('Repetitive Behaviors', 1, 10, 5)
    attention_to_detail = st.sidebar.slider('Attention to Detail', 1, 10, 5)

    # Preprocess user input for prediction
    gender_encoded = 0 if gender == 'Male' else 1
    input_data = np.array([[age, gender_encoded, communication_skills, social_interaction, repetitive_behaviors, attention_to_detail]])

    # Predict the result
    prediction = model.predict(input_data)

    # Show the prediction result
    st.subheader('Prediction Result')
    if prediction[0] == 1:
        st.write('The model predicts that the individual may have autism.')
    else:
        st.write('The model predicts that the individual is unlikely to have autism.')

if __name__ == '__main__':
    main()
