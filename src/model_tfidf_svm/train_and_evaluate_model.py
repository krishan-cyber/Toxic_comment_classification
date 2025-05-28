import nltk
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import sys
sys.path.append("..")
from data_preprocess import downSample,clean_data_text,Training_testing_split
from nltk.corpus import wordnet, stopwords 
from nltk.stem import WordNetLemmatizer
from sklearn.base import BaseEstimator, TransformerMixin
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from nltk.corpus import stopwords
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, confusion_matrix



def get_wordnet_pos(treebank_tag):

    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

class NLTKLemmatizer(BaseEstimator, TransformerMixin):
    def __init__(self, lowercase=True):
        self.lemmatizer = WordNetLemmatizer()
        self.lowercase = lowercase

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        processed_X = []
        for doc in X:
            if self.lowercase:
                doc = doc.lower()
            tokens = word_tokenize(doc)
            pos_tags = nltk.pos_tag(tokens)

            lemmatized_tokens = []
            for token, tag in pos_tags:
                wn_tag = get_wordnet_pos(tag)
                lemmatized_tokens.append(self.lemmatizer.lemmatize(token, wn_tag))
            processed_X.append(" ".join(lemmatized_tokens))
        return processed_X
    

df = pd.read_csv('../../data/Augmented_data.csv', usecols=['class', 'tweet'])
df=downSample(df,num_samples=7000)
df=clean_data_text(df)
X_train,X_test,y_train,y_test=Training_testing_split(df)

classifier_pipeline_tfIdf = make_pipeline(
    NLTKLemmatizer(lowercase=True), 
    TfidfVectorizer(ngram_range=(1,2),stop_words=stopwords.words('english')),
    OneVsRestClassifier(SVC(kernel='linear', probability=True, verbose=True, class_weight='balanced'))
)

classifier_pipeline_tfIdf.fit(X_train, y_train)
joblib.dump(classifier_pipeline_tfIdf, '../../models/model_tfidf.joblib')



y_pred = classifier_pipeline_tfIdf.predict(X_test)
print("Classification Report:\n")
print(classification_report(y_test, y_pred, target_names=["Hate Speech", "Offensive", "Normal"]))

cm = confusion_matrix(y_test, y_pred)
labels = ["Hate", "Offensive", "Normal"]
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("Confusion_matrix.png", bbox_inches='tight')
plt.close()
