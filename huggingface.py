import pandas as pd
from transformers import pipeline
from tqdm import tqdm

# Firstly split the double languages
df = pd.read_csv('preprocessed_dataset.csv')

df['native'] = df['native'].str.split('<br/>')
df = df.explode('native')
df.reset_index(drop=True, inplace=True)

# Take the top 5 languages from the native column
native_lang_count = {}
for entry in df['native']:
    if entry not in native_lang_count.keys():
        native_lang_count[entry] = 1
    else:
        native_lang_count[entry] += 1

top_5_langauges = sorted(native_lang_count, key=native_lang_count.get, reverse=True)[:5]
print(top_5_langauges)
# filtered_df = df[df['native'].isin(top_5_langauges)]

# filtered_df.to_csv("top_5_native_languages.csv", index = False)

# tqdm.pandas()

# df = pd.read_csv('preprocessed_dataset.csv')

# grammar_correction_model = pipeline(task="text2text-generation", model="hassaanik/grammar-correction-model")

# df['correct_text'] = df['text'].progress_apply(lambda x: grammar_correction_model(x, max_new_tokens=50)[0]['generated_text'])

# df.to_csv('corrected.csv', index=False)

# print(df.head())