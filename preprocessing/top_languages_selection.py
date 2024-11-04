import pandas as pd
from transformers import pipeline
from tqdm import tqdm

df = pd.read_csv('preprocessed_dataset.csv')

df['native'] = df['native'].str.split('<br/>')
df = df.explode('native')
df.reset_index(drop=True, inplace=True)

native_lang_count = {}
for entry in df['native']:
    if entry not in native_lang_count.keys():
        native_lang_count[entry] = 1
    else:
        native_lang_count[entry] += 1

top_5_langauges = sorted(native_lang_count, key=native_lang_count.get, reverse=True)[:3]

filtered_df = df[df['native'].isin(top_5_langauges)]

filtered_df.to_csv("top_3_native_languages.csv", index = False)