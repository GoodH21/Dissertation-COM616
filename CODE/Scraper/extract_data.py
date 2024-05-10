import os
import pandas as pd

# Directory containing CSV files
directory = "C:\\Users\\Harriet\\Documents\\DISSERTATION\\BB DATA SETS\\2023_Q4"

# Model number to search for
model_number = "ST12000NM0007"

# List to store DataFrames from each file
dataframes = []

print("Program is running...")

# Search for the model number in CSV files and extract data
for filename in os.listdir(directory):
    file_path = os.path.join(directory, filename)
    if os.path.isfile(file_path) and filename.endswith('.csv'):
        print(f"Searching in file: {filename}")
        # Read CSV file
        df = pd.read_csv(file_path)

        model_data = df[df.iloc[:, 2] == model_number] 
        if not model_data.empty:
            dataframes.append(model_data)

print("Search completed.")

# Combine DataFrames from all files into a single DataFrame if there are any
if dataframes:
    combined_df = pd.concat(dataframes, ignore_index=True)
    output_file = "C:\\Users\\Harriet\\Documents\\DISSERTATION\\DATA PROCESSING\\extracted_data_13.csv"
    combined_df.to_csv(output_file, index=False)  # Save combined data to CSV file
    print("Data extraction and saving completed.")
else:
    print("No data found for the specified model number.")
