import pandas as pd

# Load the dataset
file_path = 'error_types_relative_frequencies_big_dataset.csv'
df = pd.read_csv(file_path)

# Check if 'X Error' column exists and delete it
if 'X Error' in df.columns:
    df = df.drop(columns=['X Error'])
else:
    print("Column 'X Error' does not exist in the dataset.")

# Recalculate the relative frequencies for remaining error columns
# Sum all error counts for recalculating relative frequencies
error_columns = df.columns[1:]  # Assuming the first column is an identifier column
total_errors = df[error_columns].sum(axis=1)

# Recalculate relative frequencies by dividing each error count by total errors
for col in error_columns:
    df[col] = df[col] / total_errors

# Replace NaN values with 0 if there are any (in case there are zero total errors in some rows)
df = df.fillna(0)

# Save the updated DataFrame back to CSV
df.to_csv('error_types_relative_frequencies_big_dataset.csv', index=False)
print("Updated dataset saved as 'error_types_relative_frequencies_big_dataset_updated.csv'")
