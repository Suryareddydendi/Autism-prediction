import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
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
        'Sensory Sensitivity': np.random.randint(1, 10, size=100),
        'Emotional Regulation': np.random.randint(1, 10, size=100),
        'ASD Diagnosis': np.random.choice([0, 1], size=100)  # 0: No ASD, 1: ASD
    })
    return data

# Preprocess the data (encode categorical variables)
def preprocess_data(df):
    df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})  # Encode 'Gender'
    X = df.drop('ASD Diagnosis', axis=1)  # Features
    y = df['ASD Diagnosis']  # Target
    return X, y

# Train different models
def train_model(X, y, model_type='RandomForest'):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    if model_type == 'RandomForest':
        model = RandomForestClassifier(random_state=42)
    elif model_type == 'SVM':
        model = SVC()
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return model, acc

# Streamlit app layout
def main():
    st.title("Autism Prediction App")

    # Initialize the history key if it doesn't exist
    if "history" not in st.session_state:
        st.session_state["history"] = []

    # Create tabs for navigation
    tab1, tab2, tab3 = st.tabs(["Profile", "Prediction", "History"])
    
    # Tab 1: Profile
    with tab1:
        st.subheader("Profile Information")
        st.write("This app predicts autism likelihood based on several behavioral traits.")
        st.write("You can switch between tabs to use the Prediction or view your History.")
    
    # Tab 2: Prediction
    with tab2:
        st.subheader("Make a Prediction")
        
        # Load and preprocess the data
        data = load_data()
        X, y = preprocess_data(data)

        # Sidebar for model selection
        st.sidebar.header("Model Selection")
        model_type = st.sidebar.selectbox("Choose the model", ['RandomForest', 'SVM'])

        # Train model
        model, accuracy = train_model(X, y, model_type)

        # Show the accuracy of the model
        st.write(f"Model Accuracy: {accuracy * 100:.2f}%")

        # Display feature importance if RandomForest is selected
        if model_type == 'RandomForest':
            st.subheader("Feature Importance")
            feature_importance = model.feature_importances_
            feature_names = X.columns
            importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})
            importance_df = importance_df.sort_values(by='Importance', ascending=False)
            st.write(importance_df)

        # Sidebar for user input
        st.sidebar.header("Input Features")
        age = st.sidebar.slider('Age', 2, 50, 25)
        gender = st.sidebar.selectbox('Gender', ('Male', 'Female'))
        communication_skills = st.sidebar.slider('Communication Skills', 1, 10, 5)
        social_interaction = st.sidebar.slider('Social Interaction', 1, 10, 5)
        repetitive_behaviors = st.sidebar.slider('Repetitive Behaviors', 1, 10, 5)
        attention_to_detail = st.sidebar.slider('Attention to Detail', 1, 10, 5)
        sensory_sensitivity = st.sidebar.slider('Sensory Sensitivity', 1, 10, 5)
        emotional_regulation = st.sidebar.slider('Emotional Regulation', 1, 10, 5)

        # Preprocess user input for prediction
        gender_encoded = 0 if gender == 'Male' else 1
        input_data = np.array([[age, gender_encoded, communication_skills, social_interaction, repetitive_behaviors, attention_to_detail, sensory_sensitivity, emotional_regulation]])

        # Predict the result
        prediction = model.predict(input_data)

        # Show the prediction result
        st.subheader('Prediction Result')
        if prediction[0] == 1:
            result = 'The model predicts that the individual may have autism.'
        else:
            result = 'The model predicts that the individual is unlikely to have autism.'

        st.write(result)

        # Append to history after prediction
        st.session_state["history"].append({'Age': age, 'Gender': gender, 'Result': result})

    # Tab 3: History
    with tab3:
        st.subheader("Prediction History")
        if st.session_state['history']:
            history_df = pd.DataFrame(st.session_state['history'])
            st.write(history_df)
        else:
            st.write("No history available.")

if __name__ == '__main__':
    main()
