import os
import pandas as pd

def search_failures(file_path, model_failures):
    try:
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Check if the third column exists
        if len(df.columns) > 2:
            # Filter rows where the fifth column (index 4) has value 1
            failures = df[df[df.columns[4]] == 1]
            if not failures.empty:
                # Group by model name (column C) and count the number of failures for each model
                failures_by_model = failures.groupby(df.columns[2]).size().reset_index(name='Failures')
                # Update the model_failures dictionary with the current file's failures
                for _, row in failures_by_model.iterrows():
                    model = row[df.columns[2]]
                    if model in model_failures:
                        model_failures[model] += row['Failures']
                    else:
                        model_failures[model] = row['Failures']
                print(f"File: {file_path} contains 'failure' with value of 1 in column E:")
                print(failures_by_model)
                return True
            else:
                return False
        else:
            print(f"Not enough columns in file: {file_path} to search in column E")
            return False
        
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return False

# Directory containing CSV files
directory = "C:\\Users\\Harriet\\Documents\\DISSERTATION\\BB DATA SETS\\2022_Q4"

# Dictionary to store failures for each model
model_failures = {}

# Search for failures in CSV files
files_with_failures = []
for filename in os.listdir(directory):
    file_path = os.path.join(directory, filename)
    if os.path.isfile(file_path) and filename.endswith('.csv'):
        if search_failures(file_path, model_failures):
            files_with_failures.append(filename)

# Print files with failures
if files_with_failures:
    print("\nFiles with 'failure' containing value of 1 in column E:")
    for filename in files_with_failures:
        print(filename)
else:
    print("\nNo files found with 'failure' containing value of 1 in column E.")

# Print total failures for each model
print("\nTotal failures for each model:")
for model, failures in model_failures.items():
    print(f"Model: {model}, Failures: {failures}")
