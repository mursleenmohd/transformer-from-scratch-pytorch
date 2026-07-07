from datasets import load_dataset
import pandas as pd

print("Internet se dataset download ho raha hai... (Thoda wait karo)")

dataset = load_dataset("cfilt/iitb-english-hindi")

data_subset = dataset['train']['translation'][:3000]

english_sentences = [item['en'] for item in data_subset]
hindi_sentences = [item['hi'] for item in data_subset]

df = pd.DataFrame({
    'english': english_sentences,
    'hindi': hindi_sentences
})

df.to_csv('data.csv', index=False, encoding='utf-8')

print("Mubarak ho! 3000 sentences ki 'data.csv' file tumhare folder mein ban gayi hai! 🎉")