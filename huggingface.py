# Happytransformer (T5)
import pandas as pd
from happytransformer import HappyTextToText, TTSettings
from tqdm import tqdm
import re

# Load the CSV file into a DataFrame
df = pd.read_csv('top_3_native_languages.csv')

# Initialize HappyTextToText model
happy_tt = HappyTextToText("T5", "vennify/t5-base-grammar-correction")

args = TTSettings(num_beams=5, min_length=1)

def correct_grammar(text):
    sentences = re.split(r'(?<=[.!?]) +', text)
    corrected_sentences = []
    
    for sentence in sentences:
        result = happy_tt.generate_text(f"grammar: {sentence}", args=args)
        corrected_sentences.append(result.text)
    
    return ' '.join(corrected_sentences)

tqdm.pandas()

df['correct'] = df['text'].progress_apply(correct_grammar)

df.to_csv('top_3_with_corrections.csv', index=False)

print(df.head())