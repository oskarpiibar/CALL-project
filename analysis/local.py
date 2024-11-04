import pickle
import pandas as pd
from transformers import GenerationConfig  # Use the correct class for beam settings

print("LOcsl")

with open('models/model1', 'rb') as file:
    happy_tt = pickle.load(file)

beam_settings = GenerationConfig(num_beams=5, min_length=1, max_length=50)
eval1 = pd.read_csv('eval1.csv')
end = 100

learner_sentences = eval1.iloc[:end, 0]

corrected = []
for sentence in learner_sentences:
    corrected_sentence = happy_tt.generate_text(sentence, args=beam_settings).text
    corrected.append(corrected_sentence)

print(corrected)