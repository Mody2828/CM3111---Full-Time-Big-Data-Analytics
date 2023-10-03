## well_logs_app.py
import streamlit as st
import pandas as pd
import os

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
        
        # Display the dataset
        st.subheader("Dataset")
        st.write(df)
        
        

if __name__ == '__main__':
    main()
