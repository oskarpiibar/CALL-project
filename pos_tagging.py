import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag, pos_tag_sents
import pandas as pd
from tqdm import tqdm
tqdm.pandas()
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger_eng')
 
df = pd.read_csv('test_set.csv')

tagged_incorrect = pos_tag_sents(df['incorrect'].progress_apply(word_tokenize).tolist())
tagged_correct = pos_tag_sents(df['correct_text'].progress_apply(word_tokenize).tolist())

df["POS_incorrect"] = tagged_incorrect
df["POS_correct"] = tagged_correct

df.to_csv('test_set.csv', index=False)
df.head(20)

# from joblib import Parallel, delayed
# import pandas as pd
# import nltk
# from nltk.tokenize import word_tokenize
# from nltk import pos_tag

# def pos_tag_sentence(sentence):
#     return pos_tag(word_tokenize(sentence))

# df = pd.read_csv('test_set.csv')

# # Parallel processing for faster POS tagging
# num_cores = 4  # Adjust based on your CPU cores
# df['POS_incorrect'] = Parallel(n_jobs=num_cores)(delayed(pos_tag_sentence)(sent) for sent in df['incorrect'])
# df['POS_correct'] = Parallel(n_jobs=num_cores)(delayed(pos_tag_sentence)(sent) for sent in df['correct_text'])

# df.to_csv('test_set.csv', index=False)

# import spacy
# import pandas as pd

# # Load SpaCy's English model
# nlp = spacy.load("en_core_web_sm")

# # Function to POS-tag sentences using SpaCy
# def spacy_pos_tag(sentence):
#     doc = nlp(sentence)
#     return [(token.text, token.pos_) for token in doc]

# df = pd.read_csv('test_set.csv')

# # Apply POS tagging using SpaCy for both columns
# df['POS_incorrect'] = df['incorrect'].apply(spacy_pos_tag)
# df['POS_correct'] = df['correct_text'].apply(spacy_pos_tag)

# df.to_csv('test_set.csv', index=False)