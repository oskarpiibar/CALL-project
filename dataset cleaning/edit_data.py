print("AAAA")
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
import re
import ftfy
import pandas as pd
DetectorFactory.seed = 0
print("AAAA")

df_og = pd.read_csv('dataset_cleand-1-1.csv')
df = df_og.copy()

def remove_non_english(text):
    sentences = re.split(r'(?<=[.!?]) +', text)
    english_sentences = []
    for sentence in sentences:
        try:
            if detect(sentence) == 'en':  
                english_sentences.append(sentence)
        except LangDetectException:
            pass
    return ' '.join(english_sentences)

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)  
    text = text.strip()

df['text'] = df['text'].apply(ftfy.fix_text)
df['text'] = df['text'].apply(clean_text)
df['text'] = df['text'].apply(remove_non_english)


print(df.head(10))
print("AAAA")