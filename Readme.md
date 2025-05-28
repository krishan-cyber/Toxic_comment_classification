# TOXIC COMMENT CLASSIFICATION SYSTEM

Many  platforms enable individuals to share views, express opinions, and engage in discussions. However, differences in
opinion can sometimes lead to heated arguments, during which offensive language—commonly referred to as toxic comments—may arise.<br>


Toxic Comment Classification Model using Machine Learning can detect and classify toxic comments or text. 
To facilitate its use,  an API is provide and a basic user interface, allowing developers
 to integrate the model into their applications. This system empowers developers to
 incorporate toxicity detection features into their platforms, helping promote positive
 and respectful communication among users.

```bash
Toxic Comment Classification/
├── classifier_api_backend/       # Django project serving ML models via API
├── data/                         # Labeled and augmented datasets
│
├── src/                          # Source code for model training & evaluation
│   ├── data_exploration/
│   │   ├── plots/                # Output plots for visual exploration
│   │   └── data_exploration.py   # Data exploration code
│   │
│   ├── model_bert_svm/
│   │   ├── train_evaluate.py     # Training & evaluating machine learning model based on BERT and support vector machine
│   │   └── vectorizer.py         # Custom Sentence-BERT vectorizer
│   │
│   └── model_tfidf_svm/
│       └── train_and_evaluate_model.py  # Training & evaluating machine learning model based using TF-IDF and  support vector machine
│
├── data_augmentation.py          # Script to augment the dataset
└── data_preprocess.py              # Common data preprocessing utilities  
       
         
