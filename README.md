# Bank Marketing Subscription Prediction

## a. Problem Statement

The objective of this project is to build and compare multiple machine learning classification models to predict whether a customer will subscribe to a term deposit based on historical banking marketing campaign data.

The task is a binary classification problem where the target variable `y` indicates whether a customer subscribed ("yes") or not ("no"). The goal is to evaluate different machine learning models and analyze their performance using multiple evaluation metrics.

---

## b. Dataset Description

- Source: [UCI Machine Learning Repository – Bank Marketing Dataset](https://archive.ics.uci.edu/dataset/222/bank+marketing)
- Number of instances: 45,211  
- Number of input features: 16  
- Target variable: `y` (yes / no)  
- Problem type: Binary classification  

### Dataset Characteristics

- Contains both numerical and categorical features  
- No missing values  
- Imbalanced dataset (~12% positive class)  

Due to class imbalance, evaluation metrics such as AUC and MCC are particularly important for performance assessment.

---

## c. Models Used and Evaluation

The following models were trained and evaluated:

- Logistic Regression  
- Decision Tree  
- k-Nearest Neighbors (kNN)  
- Naive Bayes  
- Random Forest (Ensemble)  
- XGBoost (Ensemble)  

### Evaluation Metrics Used

Each model was evaluated using:

- Accuracy  
- AUC (Area Under ROC Curve)  
- Precision  
- Recall  
- F1 Score  
- MCC (Matthews Correlation Coefficient)  

---

## Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---------------|----------|-----|-----------|--------|----|-----|
| Logistic Regression | 0.8457 | 0.9079 | 0.4182 | 0.8147 | 0.5527 | 0.5092 |
| Decision Tree | 0.8786 | 0.6939 | 0.4800 | 0.4527 | 0.4660 | 0.3977 |
| kNN | 0.8962 | 0.8277 | 0.5990 | 0.3403 | 0.4340 | 0.4001 |
| Naive Bayes | 0.8548 | 0.8101 | 0.4059 | 0.5198 | 0.4559 | 0.3774 |
| Random Forest (Ensemble) | 0.9051 | 0.9271 | 0.6866 | 0.3478 | 0.4617 | 0.4448 |
| XGBoost (Ensemble) | 0.9055 | 0.9287 | 0.6267 | 0.4745 | 0.5401 | 0.4944 |

---

## Observations on Model Performance

| ML Model Name | Observation about model performance |
|---------------|-------------------------------------|
| Logistic Regression | Logistic Regression achieved strong recall and competitive AUC, indicating effective linear separation after preprocessing. It performs well in identifying positive cases but may produce more false positives. |
| Decision Tree | The Decision Tree model showed moderate performance but lower AUC compared to ensemble methods, indicating limited generalization ability on this dataset. |
| kNN | kNN achieved reasonable accuracy but lower AUC, suggesting sensitivity to high-dimensional feature space after one-hot encoding. |
| Naive Bayes | Naive Bayes performed moderately well but assumptions of feature independence may limit its effectiveness on correlated banking features. |
| Random Forest (Ensemble) | Random Forest achieved strong precision and robust overall performance due to ensemble averaging, reducing variance compared to a single tree. |
| XGBoost (Ensemble) | XGBoost achieved the highest AUC and strong overall performance, demonstrating superior ranking capability and effective handling of nonlinear feature interactions. |

---

## Deployment

The trained models were saved as serialized `.pkl` files and deployed using Streamlit Community Cloud.

The web application allows:

- Model selection  
- Adjustable classification threshold  
- Dataset upload for inference  
- Dynamic confusion matrix visualization  
- Evaluation metric computation  
- Business-oriented Top-X% targeting analysis  

The app is deployed [here](https://bank-marketing-classification-svwrite.streamlit.app/).

