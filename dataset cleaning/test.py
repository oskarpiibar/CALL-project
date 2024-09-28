from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
import re
import ftfy
import pandas as pd
DetectorFactory.seed = 0

df_og = pd.read_csv('dataset_cleand-1-1.csv')
df = df_og.copy()
df_1 = df.head()
print(df.head())

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

def remove_paragraph_space(text):
    return re.sub(r'\n+', '', text.strip())

def remove_emoticons(text):
    emoticon_pattern = re.compile(
        r'(:\)|:\(|\(:|\)\:|:\-\)|:\-\(|\^\^|<3|:P|:D|;\)|:-D|:O|:-O|:\')',
        flags=re.UNICODE
    )
    def replace_emoticon(match):
        end_pos = match.end()
        if end_pos < len(text) and text[end_pos] in ".!? ":
            return ''  
        else:
            return '.'  
    return emoticon_pattern.sub(replace_emoticon, text)


df_1 = df_1.drop_duplicates(subset=['text'], keep='first')
# df['text'] = df['text'].apply(ftfy.fix_text)
# df['text'] = df['text'].apply(remove_non_english)
# df = df[df['text'].str.strip().astype(bool)]

df_1.loc[:, 'text'] = df_1['text'].apply(ftfy.fix_text)
df_1.loc[:, 'text'] = df_1['text'].apply(remove_non_english)
df_1 = df_1[df_1['text'].str.strip().astype(bool)]
df_1 = df_1.reset_index(drop=True)
df_1.loc[:, 'text'] = df_1['text'].apply(remove_paragraph_space)
df_1['text'] = df_1['text'].apply(remove_emoticons)

print(df_1.head())