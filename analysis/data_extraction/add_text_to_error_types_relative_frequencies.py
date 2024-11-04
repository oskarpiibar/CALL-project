import pandas as pd

error_types_df = pd.read_csv('error_types_relative_frequencies_big_dataset.csv')
error_analysis_df = pd.read_csv('error_analysis_big_dataset_clean.csv')

if len(error_types_df) != len(error_analysis_df):
    raise ValueError("The two files do not have the same number of rows. They must match 1:1.")

text_correct_df = error_analysis_df[['text', 'correct']]

combined_df = pd.concat([error_types_df, text_correct_df], axis=1)

combined_df.to_pickle('error_types_relative_frequencies_with_text.pkl')

print("The files have been combined successfully and saved as 'error_types_relative_frequencies_with_text.csv'")
