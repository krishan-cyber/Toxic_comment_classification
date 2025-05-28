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
<br>
<br>

## Training, Testing, and Deploying the Model

### Prerequisites
1. Install Python version **3.11.0** from [python.org](https://www.python.org/downloads/).
2. Open your terminal or command palette and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Data Preparation
3. Navigate to the source directory and run data augmentation:
   ```bash
   cd src
   python data_augmentation.py
   ```

4. Run exploratory data analysis (plots will be saved in `src/data exploration/plots`):
   ```bash
   cd "src/data exploration"
   python data_exploration.py
   cd ..
   ```

### Model Training
5. Choose a model to train:

#### For BERT + SVM Model:
   ```bash
   cd src/model_bert_svm
   python train_evaluate.py
   cd ..
   ```

#### For TF-IDF + SVM Model:
   ```bash
   cd src/model_tfidf_svm
   python train_and_evaluate_model.py
   cd ..
   ```

6. Model evaluation outputs will be saved in their respective model directories.

7. Trained model files will be stored in the `models/` directory.

### Deployment
8. Move the desired trained model to the Django project for deployment:
   ```bash
   mv models/<model_file>.joblib ../classifier_api_backend/detector/
   ```

9. Start the Django server:
   ```bash
   cd ../classifier_api_backend
   python manage.py runserver
   ```

**Note:** Make sure paths are entered exactly as shown to avoid path errors.
<br>
<br>

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


       
         
