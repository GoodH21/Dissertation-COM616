import os
import pandas as pd

def search_model(file_path, model_number):
    try:
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Check if the third column exists
        if len(df.columns) > 2:
            # Filter rows where the third column (index 2) matches the model number
            model_data = df[df[df.columns[2]] == model_number]
            if not model_data.empty:
                print(f"File: {file_path} contains data for model number '{model_number}' in column C:")
                print(model_data)
                return True
            else:
                return False
        else:
            print(f"Not enough columns in file: {file_path} to search for model number")
            return False
        
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return False

# Directory containing CSV files
directory = "C:\\Users\\Harriet\\Documents\\DISSERTATION\\BB DATA SETS\\2023_Q4"

# Model number to search for
model_number = "ST12000NM0007"

# Notify that the program is running
print("Program is running...")

# Search for the model number in CSV files
files_with_model = []
for filename in os.listdir(directory):
    file_path = os.path.join(directory, filename)
    if os.path.isfile(file_path) and filename.endswith('.csv'):
        print(f"Searching for model number '{model_number}' in file: {file_path}")
        if search_model(file_path, model_number):
            files_with_model.append(filename)

# Print files with the model number
if files_with_model:
    print(f"\nFiles containing data for model number '{model_number}':")
    for filename in files_with_model:
        print(filename)
else:
    print(f"\nNo files found containing data for model number '{model_number}'.")
