import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler
from sklearn.utils import resample
import numpy as np
import joblib
from pathlib import Path

def predict_new_data(new_data, model):
    # Preprocess the new data
    X_new = new_data.drop(columns=['date']) 
    predictions = model.predict(X_new)
    return predictions

print("Loading data...")
# Define file paths for data
file_paths = ['C:\\Users\\Harriet\\Documents\\DISSERTATION\\DATA PROCESSING\\TRAINING\\Training data\\extracted_data_2019.xlsx']

# Load data from multiple files into a single DataFrame
dfs = []
for file_path in file_paths:
    if file_path.endswith('.csv'):
        dfs.append(pd.read_csv(file_path))
    elif file_path.endswith('.xlsx'):
        dfs.append(pd.read_excel(file_path))

combined_df = pd.concat(dfs, ignore_index=True)

print("Data loaded.")

print("Preparing data...")

X = combined_df.drop(columns=['failure']) 
y = combined_df['failure']

# Balance the dataset
X_nonfailures = X[y == 0]
X_failures = X[y == 1]

X_failures_resampled = resample(X_failures, replace=True, n_samples=len(X_nonfailures), random_state=42)
X_balanced = pd.concat([X_nonfailures, X_failures_resampled])
y_balanced = pd.concat([pd.Series([0]*len(X_nonfailures)), pd.Series([1]*len(X_failures_resampled))])

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.3, random_state=42)  # Adjust test_size as needed

print("Data prepared.")

# Define the model file name
model_file = 'rf_general2_trained.pkl'

# Check if the model file exists
if not Path(model_file).is_file():
    print("Training model...")
    # Define categorical and numerical features
    categorical_features = ['serial_number', 'model']
    numerical_features = ['capacity_bytes', 'smart_5_normalized', 'smart_187_normalized', 
                        'smart_188_normalized', 'smart_197_normalized', 'smart_198_normalized']

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),  
        ('scaler', StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    clf = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42))
    ])

    clf.fit(X_train, y_train)  # Train with balanced data

    # Save the trained model
    joblib.dump(clf, model_file)
    print("Model trained and saved.")
else:
    # Load the trained model
    clf = joblib.load(model_file)
    print("Model loaded.")

print("Making predictions...")
# Make predictions on the test set
y_pred = clf.predict(X_test)

print("Predictions made.")

print("Evaluating model...")
# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)
class_report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Display evaluation metrics
print("\nAccuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)
print("ROC AUC score:", roc_auc)
print("\nClassification Report:")
print(class_report)
print("\nConfusion Matrix:")
print(conf_matrix)

print("Model evaluated.")

print("Plotting confusion matrix...")
# Plot confusion matrix
def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = [0, 1]
    plt.xticks(tick_marks, ['Non-Failure', 'Failure'], rotation=45)
    plt.yticks(tick_marks, ['Non-Failure', 'Failure'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()

plot_confusion_matrix(y_test, y_pred)
plt.show()

print("Process completed.")

new_data = pd.DataFrame({
    'date': ['03/01/2019', '01/01/2020'],
    'serial_number': ['ZCH08BSC', 'ZJV0XJQ4'],
    'model': ['ST12000NM0007', 'ST12000NM0007'],
    'capacity_bytes': ['1.20E+13', '1.20E+13'],
    'smart_5_normalized': [93, 100],
    'smart_187_normalized': [78, 100],
    'smart_188_normalized': [100, 100],
    'smart_197_normalized': [100, 100],
    'smart_198_normalized': [100, 100]
})

predictions = predict_new_data(new_data, clf)
print("Predictions for new data:")
print(predictions)
