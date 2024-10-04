from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
import re
import ftfy
import pandas as pd
from tqdm import tqdm  

DetectorFactory.seed = 0

# Load the dataset
df_og = pd.read_csv('dataset_cleand-1-1.csv')
df = df_og.copy()
print(df.head(20))

# Function to remove non-English sentences
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

# Function to remove extra spaces between paragraphs
def remove_paragraph_space(text):
    return re.sub(r'\n+', '', text.strip())

# Function to remove common emoticons
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

# Drop duplicates
df = df.drop_duplicates(subset=['text'], keep='first')

# Apply text fixes with progress bars
tqdm.pandas(desc="Fixing text encoding")
df['text'] = df['text'].progress_apply(ftfy.fix_text)

tqdm.pandas(desc="Removing non-English sentences")
df['text'] = df['text'].progress_apply(remove_non_english)
df = df[df['text'].str.strip().astype(bool)].reset_index(drop=True)

tqdm.pandas(desc="Removing paragraph spaces")
df['text'] = df['text'].progress_apply(remove_paragraph_space)

tqdm.pandas(desc="Removing emoticons")
df['text'] = df['text'].progress_apply(remove_emoticons)

# Display final DataFrame
print(df.head(20))

df.to_csv('processed_dataset.csv', index=False)


