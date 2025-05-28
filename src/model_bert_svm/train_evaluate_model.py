import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sys
sys.path.append("..")
from data_preprocess import downSample,clean_data_text,Training_testing_split
from vectorizer import BERTVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix



df = pd.read_csv('../../data/Augmented_data.csv', usecols=['class', 'tweet'])
df=downSample(df,num_samples=7000)
df=clean_data_text(df)
X_train,X_test,y_train,y_test=Training_testing_split(df)



classifier_pipeline_bert = make_pipeline(
    BERTVectorizer(),
    OneVsRestClassifier(SVC(kernel='linear', probability=True, verbose=True, class_weight='balanced'))
)

classifier_pipeline_bert.fit(X_train, y_train)

joblib.dump(classifier_pipeline_bert, '../../models/model_bert.joblib')

y_pred = classifier_pipeline_bert.predict(X_test)
print("Classification Report:\n")
print(classification_report(y_test, y_pred, target_names=["Hate Speech", "Offensive", "Normal"]))

cm = confusion_matrix(y_test, y_pred)
labels = ["Hate", "Offensive", "Normal"]
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("Confusion_matrix_bert_svm.png", bbox_inches='tight')
plt.close()

