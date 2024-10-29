import pandas as pd
from collections import Counter

# Load the processed CSV file with native languages, classified errors, and error counts
df = pd.read_csv('error_analysis_by_language_clean.csv')

# Function to safely convert the error counts from string to dictionary
def convert_to_counter_dict(counter_str):
    try:
        return eval(counter_str)
    except:
        return {}

# Apply the conversion to ensure error counts are dicts
df['error_counts'] = df['error_counts'].apply(convert_to_counter_dict)

# Convert the error counts into a dataframe for easier plotting
error_types_df = pd.DataFrame(df['error_counts'].tolist(), index=df['native']).fillna(0)

# Normalize data to ensure consistent data types
error_types_df = error_types_df.applymap(lambda x: int(x) if isinstance(x, (int, float)) else 0)

# Calculate relative frequencies by dividing each error count by the total number of errors for that language
error_types_relative_df = error_types_df.div(error_types_df.sum(axis=1), axis=0)

error_types_relative_df.to_pickle('error_types_relative_frequencies.pkl')

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


