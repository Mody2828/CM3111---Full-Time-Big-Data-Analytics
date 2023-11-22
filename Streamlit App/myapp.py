# well_logs_app.py
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
import os
import numpy as np
import matplotlib.pyplot as plt

# Load dataset function
@st.cache_data
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

# Main function
def main():
    st.title("Well Logs Dataset Exploration")

    # I will be using multiple datasets files
    # Get a list of CSV files in the current directory
    files = [file for file in os.listdir() if file.endswith(".csv")]

    # To choose a dataset file
    uploaded_file = st.selectbox("Choose a dataset file", files)

    if uploaded_file is not None:
        # Read the dataset
        df = load_data(uploaded_file)

        # Display the Dataset Information
        st.subheader("Dataset Information")
        st.write(f"Dataset ID: {uploaded_file}")
        st.write(f"Number of rows: {df.shape[0]}")
        st.write(f"Number of columns: {df.shape[1]}")

        # Get unique well IDs from selected file
        wells = df['Well_ID'].unique()

        # To choose a well ID
        uploaded_well = st.selectbox("Choose a well ID", wells)

        # Filter dataframe based on selected well ID
        if uploaded_well is not None:
            
            well_df = df[df['Well_ID'] == uploaded_well]
            

            # Get data for all other wells
            other_wells_df = df[df['Well_ID'] != uploaded_well]

            # Choose a classifier
            classifier = st.selectbox("Choose a classifier", ["SVM", "Random Forest"])

            if classifier == "SVM":
                # Train an SVM classifier on other wells
                st.write("Training SVM Classifier...")
                X = other_wells_df[['Depth', 'GAM(NAT)', 'RES(16N)', 'RES(64N)', 'NEUTRON']]
                y = other_wells_df['Interpretation']
                svm_classifier = SVC()
                svm_classifier.fit(X, y)

                # Test the model on the selected well
                X_test = well_df[['Depth', 'GAM(NAT)', 'RES(16N)', 'RES(64N)', 'NEUTRON']]
                y_test = well_df['Interpretation']
                y_pred = svm_classifier.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                st.write(f"SVM Classifier Accuracy: {accuracy}")

                # Visualization
                st.subheader("Outcome Visualization")
                visualize_outcome(x=well_df['Depth'], y_actual=y_test, y_predicted=y_pred)

            elif classifier == "Random Forest":
                # Train a Random Forest classifier on other wells
                st.write("Training Random Forest Classifier...")
                X = other_wells_df[['Depth', 'GAM(NAT)', 'RES(16N)', 'RES(64N)', 'NEUTRON']]
                y = other_wells_df['Interpretation']
                rf_classifier = RandomForestClassifier()
                rf_classifier.fit(X, y)

                # Test the model on the selected well
                X_test = well_df[['Depth', 'GAM(NAT)', 'RES(16N)', 'RES(64N)', 'NEUTRON']]
                y_test = well_df['Interpretation']
                y_pred = rf_classifier.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                st.write(f"Random Forest Classifier Accuracy: {accuracy}")

                # Visualization
                st.subheader("Outcome Visualization")
                visualize_outcome(x=well_df['Depth'], y_actual=y_test, y_predicted=y_pred)

# Function to visualize outcome
def visualize_outcome(x, y_actual, y_predicted):
    le = preprocessing.LabelEncoder()

    # Ploting the Actual Outcome and Predicted Outcome Vs well Depth
    y_actual_decode = le.fit_transform(y_actual)
    y_predicted_decode = le.fit_transform(y_predicted)

    # Create a smoother curve by polynomial interpolation
    x_smooth = np.linspace(x.min(), x.max(), 300)
    p_actual = np.polyfit(x, y_actual_decode, 3)
    p_predicted = np.polyfit(x, y_predicted_decode, 3)

    y_actual_smooth = np.polyval(p_actual, x_smooth)
    y_predicted_smooth = np.polyval(p_predicted, x_smooth)

    # Plot the smoothed curves
    fig, ax = plt.subplots()
    ax.plot(x_smooth, y_actual_smooth, label='Actual Outcome')
    ax.plot(x_smooth, y_predicted_smooth, label='Predicted Outcome')
    ax.set_ylim(0, 2)  # Adjusted y-axis limits
    ax.grid()
    ax.legend()

    # Display the plot using Streamlit
    st.pyplot(fig)

if __name__ == '__main__':
    main()
