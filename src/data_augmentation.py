import pandas as pd
from googletrans import Translator
import random


df = pd.read_csv('../data/labeled_data.csv', usecols=['class', 'tweet'])
df_hate = df[df['class'] == 0]
df_offensive = df[df['class'] == 1]
df_normal = df[df['class'] == 2]

translator = Translator()
augmented_texts = []
languages = ['fr', 'de', 'es', 'ja', 'zh-cn']
for text in df_hate['tweet'].sample(frac=0.3, random_state=42): 
    try:
        lang = random.choice(languages)
        translated = translator.translate(text, dest=lang).text
        back_translated = translator.translate(translated, dest='en').text
        augmented_texts.append(back_translated)
    except Exception as e:
        print("Translation error:", e)
        continue


aug_df = pd.DataFrame({'class': 0, 'tweet': augmented_texts})
new_df = pd.concat([df, aug_df], ignore_index=True)
new_df.to_csv("../data/Augmented_data.csv")