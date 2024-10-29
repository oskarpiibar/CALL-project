import pandas as pd
import numpy as np
print(np.__version__)


# Load the dataset
data = pd.read_csv('small.csv')

# Drop the 'error_counts' column
data = data.drop(columns=['error_counts'])

# Save the updated dataset back to the CSV file
data.to_csv('for_error_position_small.csv', index=False)

print("The 'error_counts' column has been removed.")
