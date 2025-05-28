import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.utils import resample


def downSample(df,num_samples=7000):
    df_hate = df[df['class'] == 0]
    df_offensive = df[df['class'] == 1]
    df_normal = df[df['class'] == 2]
    df_offensive_downsampled = resample(df_offensive,
                                    replace=False,
                                    n_samples=num_samples,
                                    random_state=42)
    df_balanced = pd.concat([df_hate, df_offensive_downsampled, df_normal])
    df = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
    return df

def clean_data_text(df):
    def clean_text(text):
        return re.sub(r'[^a-zA-Z]+', ' ', text.lower())
    df['clean_tweet'] = df['tweet'].apply(clean_text)
    return df


def Training_testing_split(df):
    X_train, X_test, y_train, y_test = train_test_split(
    df['clean_tweet'], df['class'], test_size=0.2, stratify=df['class'], random_state=42
)   
    return X_train, X_test, y_train, y_test 