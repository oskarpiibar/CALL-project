import pandas as pd
from transformers import pipeline

# Load the CSV file into a DataFrame
df = pd.read_csv('preprocessed_dataset.csv')

# Load the correction model from Hugging Face
grammar_correction_model = pipeline(task="text2text-generation", model="hassaanik/grammar-correction-model")

# Apply text correction to the 'text' column
df['correct_text'] = df['text'].apply(lambda x: grammar_correction_model(x)[0]['generated_text'])

# Save the corrected dataset to a new CSV file
df.to_csv('corrected.csv', index=False)
