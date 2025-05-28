import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
import re
import string
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from collections import Counter


df = pd.read_csv('../../data/labeled_data.csv', usecols=['class', 'tweet'])

print("Shape:", df.shape)
print(df.info())

print("Missing values:\n", df.isnull().sum())
print("Duplicate rows:", df.duplicated().sum())
df = df.drop_duplicates()

category = {0: "Hate Speech", 1: "Offensive Language", 2: "Normal"}
df['label'] = df['class'].map(category)
df['label'].value_counts()

plt.figure(figsize=(8,5))
sns.countplot(data=df, x='label', palette='pastel')
plt.title('Class Distribution')
plt.xlabel('Label')
plt.ylabel('Count')
plt.xticks(rotation=15)
plt.savefig("plots/Class_Distribution.png", bbox_inches='tight')
plt.close()

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text) 
    text = re.sub(r"[^a-z\s]", "", text) 
    text = re.sub(r"\s+", " ", text).strip() 
    return text

df['clean_tweet'] = df['tweet'].apply(clean_text)

df['text_len'] = df['clean_tweet'].apply(len)

plt.figure(figsize=(8,5))
sns.histplot(df['text_len'], bins=30, kde=True, color='skyblue')
plt.title("Tweet Length Distribution")
plt.xlabel("Length of tweet")
plt.ylabel("Frequency")
plt.savefig("plots/Tweet_Length_Distribution.png", bbox_inches='tight')
plt.close()


plt.figure(figsize=(8,5))
sns.boxplot(x='label', y='text_len', data=df, palette='Set2')
plt.title("Tweet Length by Class")
plt.xlabel("Label")
plt.ylabel("Tweet Length")
plt.savefig("plots/Tweet_Length_by_Class.png", bbox_inches='tight')
plt.close()


stop_words = set(stopwords.words('english'))

def generate_wordcloud(class_value, title):
    text = " ".join(df[df['class'] == class_value]['clean_tweet'])
    wordcloud = WordCloud(stopwords=stop_words, background_color='white',
                          width=800, height=400).generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(title, fontsize=16)
    plt.axis('off')
    plt.savefig(f"plots/{title}.png", bbox_inches='tight')
    plt.close()


generate_wordcloud(0, "Hate Speech WordCloud")
generate_wordcloud(1, "Offensive Language WordCloud")
generate_wordcloud(2, "Normal Text WordCloud")


def most_common_words(class_value, n=15):
    text = " ".join(df[df['class'] == class_value]['clean_tweet'])
    words = [word for word in text.split() if word not in stop_words]
    common = Counter(words).most_common(n)
    words, counts = zip(*common)
    
    plt.figure(figsize=(8, 5))
    sns.barplot(x=list(counts), y=list(words), palette='magma')
    plt.title(f"Top {n} Words - {category[class_value]}")
    plt.xlabel("Count")
    plt.ylabel("Word")
    plt.savefig(f"plots/Top {n} Words - {category[class_value]}.png", bbox_inches='tight')
    plt.close()


most_common_words(0)
most_common_words(1)
most_common_words(2)