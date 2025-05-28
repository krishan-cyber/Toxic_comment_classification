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
```
## Training,Testing,Deploying model on your own
1. Download python version==3.11.0 from https://www.python.org/downloads/
2. open command pallete and run "pip install -r requirements.txt"
3. run on command pallete "cd src" then "python data_augmentation.py"
4. run cd "src/data exploration" then python python data_exploartion.py to exploartory data analysis out will be found at src/data exploration/plots
5. choose model to train
6.  for model_bert_svm   run cd src/model_bert_svm and then python train_evaluate_python.py
7.  for model_tfidf run cd src/model_tfidf_svm and then python train_and_evaluate_model.py
8.  model evaluation data will be found at respective model directory
9.  trained model can be found at models/
10.  move trained model to classifier_api_backend/detector
11.  run the django server







## This project also provides Api
## API Usage Instructions

**Endpoint:** `/api/predict/`  
**Method:** `POST`  
**Content-Type:** `application/json`

### Request Body (JSON)
```json
{
  "text": "I hate you!"
}
```

### Response Body (JSON)
```json
{
  "prediction": 1,
  "class_name": "Offensive Language",
  "probabilities": {
    "Hate Speech": 0.05,
    "Offensive Language": 0.91,
    "Normal": 0.04
  }
}
```

### Class Labels
- `0` = Hate Speech  
- `1` = Offensive Language  
- `2` = Normal  

Note: This API only supports English text.


       
         
