import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

# Load models
@st.cache_resource
def load_models():
    models = {
        'GRU': load_model('GRU_model.h5'),
        'LSTM': load_model('LSTM_model.h5'),
        'Bidirectional-LSTM': load_model('Bidirectional-LSTM_model.h5')
    }
    return models

models = load_models()

# Load dataset
@st.cache_data
def load_data():
    data = pd.read_csv('balanced_creditcard_data.csv')
    return data

data = load_data()

# Data preprocessing
X = data.drop('Class', axis=1).values
y = data['Class'].values
X = np.reshape(X, (X.shape[0], 1, X.shape[1]))

# Evaluate models
def evaluate_model(model, X_test, y_test):
    y_pred = (model.predict(X_test) > 0.5).astype("int32")
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    return accuracy, precision, recall, cm

# Display performance metrics
def display_confusion_matrix():
    st.title('Credit Card Fraud Comparative Analysis')
    st.subheader('Confusion Matrix for Each Model')
    
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    for name, model in models.items():
        _, _, _, cm = evaluate_model(model, X_test, y_test)
        
        st.subheader(f'{name} Confusion Matrix')
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Fraud', 'Fraud'], yticklabels=['Non-Fraud', 'Fraud'], ax=ax)
        plt.title(f'{name} Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        st.pyplot(fig)

def display_comparative_results():
    st.title('Credit Card Fraud Comparative Analysis')
    st.subheader('Comparative Analysis of Model Performance')
    
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    metrics = []
    for name, model in models.items():
        accuracy, precision, recall, _ = evaluate_model(model, X_test, y_test)
        metrics.append([name, accuracy, precision, recall])
    
    metrics_df = pd.DataFrame(metrics, columns=['Model', 'Accuracy', 'Precision', 'Recall'])
    
    st.subheader('Model Performance Metrics')
    st.dataframe(metrics_df.set_index('Model').round(10))  # Display table with 10 decimal points
    
    # Line graph for metrics
    st.subheader('Performance Metrics Line Graph')
    metrics_df.set_index('Model').plot(kind='line', marker='o', figsize=(12, 8))
    plt.title('Performance Metrics')
    plt.ylabel('Score')
    plt.xticks(rotation=0)
    st.pyplot(plt.gcf())
    
    # Best model
    best_model = metrics_df.loc[metrics_df['Accuracy'].idxmax()]
    st.subheader('Best Model')
    st.write(f"The best model based on accuracy is {best_model['Model']} with an accuracy of {best_model['Accuracy']:.10f}.")

# Streamlit app with drop-down menu
def main():
    st.sidebar.title('Navigation')
    choice = st.sidebar.selectbox('Select an Option', ['Confusion Matrix', 'Comparative Results'])
    
    if choice == 'Confusion Matrix':
        display_confusion_matrix()
    elif choice == 'Comparative Results':
        display_comparative_results()

if __name__ == "__main__":
    main()
