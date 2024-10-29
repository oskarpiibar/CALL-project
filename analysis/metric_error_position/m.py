import pandas as pd

# Load the dataset
data = pd.read_csv('intermediate_results.csv')

# Drop the 'error_counts' column
data = data.drop(columns=['error_counts'])

# Save the updated dataset back to the CSV file
data.to_csv('for_error_position_big.csv', index=False)

print("The 'error_counts' column has been removed.")
