import pandas as pd
import torch
from sklearn.base import BaseEstimator, TransformerMixin
from sentence_transformers import SentenceTransformer

class BERTVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, model_name='all-MiniLM-L6-v2', device=None):
        self.model_name = model_name 
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = SentenceTransformer(model_name, device=self.device)
        
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.Series):
            X = X.tolist()
        return self.model.encode(X, convert_to_tensor=True, device=self.device).cpu().numpy()