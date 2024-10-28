import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Load the CSV file (without the 'native' column)
df = pd.read_csv('error_analysis_big_dataset_clean.csv')
df = df.head(50)

# Function to safely convert the error counts from string to dictionary
def convert_to_counter_dict(counter_str):
    try:
        return eval(counter_str)
    except:
        return {}

# Apply the conversion to ensure error counts are dicts
df['error_counts'] = df['error_counts'].apply(convert_to_counter_dict)

# Convert the error counts into a DataFrame for easier plotting
error_types_df = pd.DataFrame(df['error_counts'].tolist()).fillna(0)

# Normalize data to ensure consistent data types
error_types_df = error_types_df.applymap(lambda x: int(x) if isinstance(x, (int, float)) else 0)

# Calculate relative frequencies by dividing each error count by the total number of errors for that row
error_types_relative_df = error_types_df.div(error_types_df.sum(axis=1), axis=0)

# Save the relative frequencies DataFrame to a pickle file
error_types_relative_df.to_pickle('error_types_relative_frequencies_big_dataset.pkl')
error_types_relative_df.to_csv('error_types_relative_frequencies_big_dataset.csv')

# # Plot the stacked bar chart for absolute error counts (plt.figure() removed here)
# error_types_df.plot(kind='bar', stacked=True, colormap='tab20', figsize=(12, 8))
# plt.title("Error Types Across Native Languages (Absolute Counts)")
# plt.xlabel("Native Language")
# plt.ylabel("Number of Errors")
# plt.legend(title="Error Types", bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.xticks(rotation=45, ha='right')
# plt.tight_layout()

# # Show plot for absolute counts
# plt.show()

# # Plot the stacked bar chart for relative frequencies (plt.figure() removed here)
# error_types_relative_df.plot(kind='bar', stacked=True, colormap='tab20', figsize=(12, 8))
# plt.title("Error Types Across Native Languages (Relative Frequencies)")
# plt.xlabel("Native Language")
# plt.ylabel("Relative Frequency")
# plt.legend(title="Error Types", bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.xticks(rotation=45, ha='right')
# plt.tight_layout()

# # Show plot for relative frequencies
# plt.show()


