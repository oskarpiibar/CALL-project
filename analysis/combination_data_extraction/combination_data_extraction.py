import pandas as pd

df1 = pd.read_csv('../metric_error_position/classified_texts_Russian_95.csv')[['text', 'correct']].drop_duplicates()
df2 = pd.read_csv('../metric_rel_frequency/classified_texts_Russian_90.csv')[['text', 'correct']].drop_duplicates()
df3 = pd.read_csv('../metric_pos_sequences/sequence_extracted_texts_Russian_4.csv')[['text', 'correct']].drop_duplicates()

df4 = pd.read_csv('../metric_error_position/classified_texts_Spanish_90.csv')[['text', 'correct']].drop_duplicates()
df5 = pd.read_csv('../metric_rel_frequency/classified_texts_Spanish_90.csv')[['text', 'correct']].drop_duplicates()
df6 = pd.read_csv('../metric_pos_sequences/sequence_extracted_texts_Spanish_4.csv')[['text', 'correct']].drop_duplicates()

df7 = pd.read_csv('../metric_error_position/classified_texts_Chinese (Mandarin)_90.csv')[['text', 'correct']].drop_duplicates()
df8 = pd.read_csv('../metric_rel_frequency/classified_texts_Chinese (Mandarin)_90.csv')[['text', 'correct']].drop_duplicates()
df9 = pd.read_csv('../metric_pos_sequences/sequence_extracted_texts_Chinese (Mandarin)_4.csv')[['text', 'correct']].drop_duplicates()

combined_Russian_df = pd.concat([df1, df2, df3]).drop_duplicates()
combined_Spanish_df = pd.concat([df4, df5, df6]).drop_duplicates()
combined_Chinese_df = pd.concat([df7, df8, df9]).drop_duplicates() 

combined_Russian_df.to_csv('combined_Russian.csv', index=False)
combined_Spanish_df.to_csv('combined_Spanish.csv', index=False)
combined_Chinese_df.to_csv('combined_Chinese.csv', index=False) 

print("The files have been combined and saved without duplicates.")
