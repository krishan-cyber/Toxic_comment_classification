import joblib
import os
import __main__
from .vectorizer import BERTVectorizer
setattr(__main__,'BERTVectorizer',BERTVectorizer)

model_path = os.path.join(os.path.dirname(__file__), 'model_bert.joblib')
model = joblib.load(model_path)

def predict(text):
    return model.predict([text])[0]
def predict_full(text):
    return model.predict_proba([text])[0]
