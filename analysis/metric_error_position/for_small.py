import pandas as pd
from tqdm import tqdm
import ast

data = pd.read_csv('for_error_position_little.csv')

tqdm.pandas()

data['original_pos'] = data['original_pos'].apply(eval)

data = data[data['classified_errors'] != '[]']

data['classified_errors'] = data['classified_errors'].apply(lambda x: ast.literal_eval(x) if x != '[]' else [])


def precise_classify_error_position(row):
    tokens = row['original_pos']
    errors = row['classified_errors']
    length = len(tokens)

    begin_threshold = int(0.2 * length)
    end_threshold = length - begin_threshold

    position_count = {'beginning': 0, 'middle': 0, 'end': 0}
    
    for error in errors:
        error_index = None
        for i, (word, pos) in enumerate(tokens):
            if word == error[1]:  
                error_index = i
                break

        if error_index is not None:
            if error_index < begin_threshold:
                position_count['beginning'] += 1
            elif error_index >= end_threshold:
                position_count['end'] += 1
            else:
                position_count['middle'] += 1

    total_errors = sum(position_count.values())
    if total_errors > 0:
        for position in position_count:
            position_count[position] = (position_count[position] / total_errors) * 100
    else:
        position_count = {'beginning': 0, 'middle': 0, 'end': 0}
    
    return position_count

data[['error_beginning_pct', 'error_middle_pct', 'error_end_pct']] = data.progress_apply(
    lambda row: pd.Series(precise_classify_error_position(row)), axis=1
)

error_position_summary = data.groupby('native')[['error_beginning_pct', 'error_middle_pct', 'error_end_pct']].mean()

error_position_summary.to_csv('ground.csv')

print("The error position percentage data has been saved to 'error_position_percentage_by_native.csv'")
