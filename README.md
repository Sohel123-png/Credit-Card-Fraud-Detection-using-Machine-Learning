# Credit Card Fraud Detection Project

## Project Overview
This project aims to build a machine learning model to detect fraudulent credit card transactions. Due to the highly imbalanced nature of the dataset, where fraudulent transactions are very rare, special techniques were used to train a robust classifier. This notebook covers the entire workflow from data exploration and visualization to model training, evaluation, and hyperparameter tuning.

---

## Dataset
The dataset used is the "Credit Card Fraud Detection" dataset from Kaggle. It contains anonymized transaction data from European cardholders over two days.

- **Source:** [Kaggle Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/credit-card-fraud-detection)
- **Features:** The dataset includes 28 principal components (`V1` to `V28`), along with `Time` and `Amount`. The target variable is `Class` (1 for fraud, 0 for legitimate).

---

## Methodology
The project follows these key steps:
1.  **Exploratory Data Analysis (EDA):** Visualized the class imbalance and analyzed the distributions of 'Time' and 'Amount' features for both fraudulent and legitimate transactions.
2.  **Preprocessing:**
    - Scaled the `Time` and `Amount` columns using `StandardScaler` to bring them to a common scale.
    - Split the original, imbalanced dataset into training and testing sets to ensure the model is evaluated on a realistic data distribution.
3.  **Handling Class Imbalance:** Applied the **SMOTE (Synthetic Minority Over-sampling Technique)** on the training data to create a balanced set for the models to learn from. This prevents the model from being biased towards the majority class.
4.  **Model Training & Comparison:** Trained several baseline models (Logistic Regression, Random Forest) on the resampled data and evaluated them using metrics suitable for imbalanced datasets like Precision, Recall, and F1-Score.
5.  **Hyperparameter Tuning:** Optimized the best-performing model (Random Forest) using `GridSearchCV` to find the most effective parameters.
6.  **Final Evaluation:** The tuned model was evaluated on the unseen, imbalanced test set, and its performance was visualized using a confusion matrix and an ROC curve.

---

## Results
After training and tuning, the models showed the following performance on the test set. The optimized Random Forest model demonstrated the best balance between precision and recall.

*<-- यहाँ पर आप अपने Jupyter Notebook से Results वाला टेबल कॉपी-पेस्ट करें -->*
| Model | Accuracy | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- | :--- |
| Logistic Regression | 0.973 | 0.057 | 0.912 | 0.107 |
| Random Forest | 0.999 | 0.814 | 0.824 | 0.819 |
| **Optimized RF** | **0.999** | **0.840** | **0.831** | **0.835** |



---

## How to Run
1.  Clone this repository to your local machine.
2.  Install the required libraries using pip:
    ```bash
    pip install -r requirements.txt
    ```
3.  Open the Jupyter Notebook `credit_card_fraud_detection.ipynb` and run the cells.

---

## Libraries Used
- pandas
- numpy
- scikit-learn
- imbalanced-learn
- seaborn
- matplotlib