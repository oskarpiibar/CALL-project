import pandas as pd

# Load the three CSV files
df1 = pd.read_csv('../metric_error_position/classified_texts_Russian.csv')
df2 = pd.read_csv('../metric_rel_frequency/classified_texts_Russian.csv.csv')
df3 = pd.read_csv('file3.csv')

df4 = pd.read_csv('../metric_error_position/classified_texts_Spanish.csv')
df5 = pd.read_csv('../metric_rel_frequency/classified_texts_Spanish.csv')
df6 = pd.read_csv('../-/classified_texts_Spanish.csv')

df7 = pd.read_csv('../metric_error_position/classified_texts_Chinese (Mandarin).csv')
df8 = pd.read_csv('../metric_rel_frequency/classified_texts_Chinese (Mandarin).csv')
df9 = pd.read_csv('../-/classified_texts_Chinese (Mandarin).csv')



# Concatenate the dataframes and drop duplicate rows
combined_Russian_df = pd.concat([df1, df2, df3]).drop_duplicates()
combined_Spanish_df = pd.concat([df4, df5, df6]).drop_duplicates()
combined_Chinese_df = pd.concat([df7, df8, df9]).drop_duplicates()

# Save the combined DataFrame to a new CSV file
combined_Russian_df.to_csv('combined_Russian.csv', index=False)
combined_Spanish_df.to_csv('combined_Spanish.csv', index=False)
combined_Chinese_df.to_csv('combined_Chinese.csv', index=False)


print("The files have been combined and saved without duplicates.")
