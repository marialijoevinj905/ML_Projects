import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

@st.cache_data
def load_data():
    column_names = [
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
        'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
    ]
    df = pd.read_csv('heart_disease_uci.csv', header=None, names=column_names)
    df = df.replace('?', np.nan)
    df = df.dropna()
    return df

def preprocess_data(df):
    X = df.iloc[:, :-1]  # Features
    y = df.iloc[:, -1]   # Target variable
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = X.select_dtypes(exclude=['object', 'category']).columns.tolist()
    
    # Create a preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(), categorical_cols)
        ]
    )
    
    # Fit and transform the data
    X_processed = preprocessor.fit_transform(X)
    return X_processed, y, numerical_cols, categorical_cols, preprocessor

def train_and_evaluate_models(X_train, X_test, y_train, y_test, selected_models):
    models = {
        'Logistic Regression': LogisticRegression(max_iter=200),
        'Support Vector Machine': SVC(),
        'Decision Tree': DecisionTreeClassifier(),
        'Random Forest': RandomForestClassifier()
    }
    
    results = {}
    
    for model_name in selected_models:
        model = models[model_name]
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        results[model_name] = (accuracy, predictions)
        
    return results

# Streamlit app
st.title("Heart Disease Prediction App")

df = load_data()
X, y, numerical_cols, categorical_cols, preprocessor = preprocess_data(df)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

st.header("Model Performance Comparison")

# Model selection
selected_models = st.multiselect("Select Models", 
    ['Logistic Regression', 'Support Vector Machine', 'Decision Tree', 'Random Forest'])

# Initialize results
results = {}

# Train and evaluate the selected models
if selected_models:
    results = train_and_evaluate_models(X_train, X_test, y_train, y_test, selected_models)

# Accuracy comparison and visualization
if results:  # Check if results is not empty
    st.subheader("Accuracy Comparison of Selected Models")
    
    results_df = pd.DataFrame.from_dict(
        {model_name: acc for model_name, (acc, _) in results.items()},
        orient='index', columns=['Accuracy']
    ).reset_index().rename(columns={'index': 'Model'})

    st.write(results_df)

    # Bar Plot of Model Accuracies
    st.subheader("Model Accuracy Comparison")
    fig, ax = plt.subplots()
    sns.barplot(x='Accuracy', y='Model', data=results_df, ax=ax)
    ax.set_title("Model Accuracy Comparison")
    st.pyplot(fig)
else:
    st.write("No models selected or evaluated.")

# Prediction section
st.header("Make Predictions")

# Create sliders for input data
age = st.slider("Input value for age", min_value=0, max_value=120, value=30)
sex = st.slider("Input value for sex (0=Female, 1=Male)", min_value=0, max_value=1, value=1)
cp = st.slider("Input value for cp (chest pain type)", min_value=0, max_value=3, value=0)
trestbps = st.slider("Input value for trestbps (resting blood pressure)", min_value=0, value=120)
chol = st.slider("Input value for chol (serum cholesterol)", min_value=0, value=200)
fbs = st.slider("Input value for fbs (fasting blood sugar)", min_value=0, max_value=1, value=0)
restecg = st.slider("Input value for restecg (resting electrocardiographic results)", min_value=0, max_value=2, value=0)
thalach = st.slider("Input value for thalach (maximum heart rate achieved)", min_value=0, value=150)
exang = st.slider("Input value for exang (exercise induced angina)", min_value=0, max_value=1, value=0)
oldpeak = st.slider("Input value for oldpeak (depression induced by exercise)", min_value=0.0, value=0.0)
slope = st.slider("Input value for slope (slope of the peak exercise ST segment)", min_value=0, max_value=2, value=0)
ca = st.slider("Input value for ca (number of major vessels)", min_value=0, max_value=4, value=0)
thal = st.slider("Input value for thal (thalassemia)", min_value=0, max_value=3, value=0)

# Prepare input data for prediction
input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])

# Create a DataFrame for input data to match the preprocessor's expected format
input_data_df = pd.DataFrame(input_data, columns=numerical_cols + categorical_cols)

if st.button("Predict"):
    # Select the last selected model for prediction
    selected_model_for_prediction = selected_models[-1] if selected_models else None
    if selected_model_for_prediction:
        model = {
            'Logistic Regression': LogisticRegression(max_iter=200),
            'Support Vector Machine': SVC(),
            'Decision Tree': DecisionTreeClassifier(),
            'Random Forest': RandomForestClassifier()
        }[selected_model_for_prediction]

        # Fit the model on the training data for prediction
        model.fit(X_train, y_train)

        # Transform the input data using the preprocessor
        input_data_processed = preprocessor.transform(input_data_df)

        # Make prediction
        prediction = model.predict(input_data_processed)

        if prediction[0] == 1:
            st.success("Prediction: Heart Disease Present")
        else:
            st.success("Prediction: No Heart Disease")
