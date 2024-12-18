import pandas as pd
from tqdm import tqdm
import ast

# Load the dataset
data = pd.read_csv('for_error_position_big.csv')

# Initialize tqdm with pandas apply
tqdm.pandas()

# Convert 'original_pos' to list format if stored as a string representation
data['original_pos'] = data['original_pos'].apply(eval)

# Filter out rows where 'classified_errors' is empty
data = data[data['classified_errors'] != '[]']

# Convert 'classified_errors' to list format if stored as a string representation
data['classified_errors'] = data['classified_errors'].apply(lambda x: ast.literal_eval(x) if x != '[]' else [])

def precise_classify_error_position(row):
    # Token list for the original sentence
    tokens = row['original_pos']
    errors = row['classified_errors']
    length = len(tokens)

    # Calculate boundaries for beginning, middle, and end
    begin_threshold = int(0.2 * length)
    end_threshold = length - begin_threshold

    # Initialize counters
    position_count = {'beginning': 0, 'middle': 0, 'end': 0}
    
    for error in errors:
        error_index = None
        # Locate the error word in the token list
        for i, (word, pos) in enumerate(tokens):
            if word == error[1]:  # assuming error[1] is the affected word
                error_index = i
                break

        # Classify position with refined boundaries
        if error_index is not None:
            if error_index < begin_threshold:
                position_count['beginning'] += 1
            elif error_index >= end_threshold:
                position_count['end'] += 1
            else:
                position_count['middle'] += 1

    # Calculate total errors for percentage calculation
    total_errors = sum(position_count.values())
    if total_errors > 0:
        for position in position_count:
            position_count[position] = (position_count[position] / total_errors) * 100
    else:
        position_count = {'beginning': 0, 'middle': 0, 'end': 0}
    
    return position_count

# Apply the function with a progress bar
data[['error_beginning_pct', 'error_middle_pct', 'error_end_pct']] = data.progress_apply(
    lambda row: pd.Series(precise_classify_error_position(row)), axis=1
)

# Select columns to save: 'text', 'correct', and the calculated percentages
output_data = data[['text', 'correct', 'error_beginning_pct', 'error_middle_pct', 'error_end_pct']]

# Save to CSV
output_data.to_csv('to_classify.csv', index=False)

print("The error position percentage data has been saved to 'to_classify.csv'")
