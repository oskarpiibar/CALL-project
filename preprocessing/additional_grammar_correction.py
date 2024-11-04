import pandas as pd
from tqdm import tqdm
import re
from transformers import pipeline

tqdm.pandas()

df = pd.read_csv('top_3_with_corrections.csv')

grammar_correction_model = pipeline(task="text2text-generation", model="hassaanik/grammar-correction-model")

def grammar_correction(text):
    sentences = re.split(r'(?<=[.!?]) +', text)
    corrected_sentences = []

    for sentence in sentences:
        correct_sentence = grammar_correction_model(sentence)[0]['generated_text']
        corrected_sentences.append(correct_sentence)

    return ' '.join(corrected_sentences)


df['correction_v2'] = df['correct'].progress_apply(grammar_correction)

df.to_csv('additional_grammar_correction.csv', index=False)

print(df.head())