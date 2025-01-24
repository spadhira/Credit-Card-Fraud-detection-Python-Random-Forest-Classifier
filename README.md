# Credit Card Fraud Detection

This project focuses on detecting fraudulent credit card transactions using machine learning techniques. It addresses the challenge of identifying fraudulent transactions in a highly imbalanced dataset, where the number of fraudulent cases is significantly smaller than valid transactions. The solution employs advanced data preprocessing, model training, and evaluation techniques to ensure reliable and accurate predictions.

## Features

- **Data Preprocessing**:
  - Handled missing values by replacing them with median values.
  - Analyzed and visualized the distribution of fraudulent vs. valid transactions.
  - Generated a correlation heatmap to identify feature relationships.

- **Class Imbalance Handling**:
  - Utilized SMOTE (Synthetic Minority Oversampling Technique) to balance the dataset by oversampling the minority class.

- **Machine Learning Model**:
  - Trained a Random Forest Classifier to classify transactions as fraudulent or valid.
  - Evaluated model performance using metrics such as:
    - Accuracy
    - Precision
    - Recall
    - F1-score
    - Matthews Correlation Coefficient (MCC)
    - Confusion Matrix
    - ROC-AUC Curve

- **Visualization**:
  - Plotted the confusion matrix for detailed performance analysis.
  - Created the ROC-AUC curve to evaluate the classifier's discrimination ability.

## Dataset

The dataset used for this project is sourced from [Kaggle - Credit Card Fraud Detection Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud). It contains the following key features:
- **Time**: Seconds elapsed between each transaction and the first transaction in the dataset.
- **V1-V28**: Anonymized features derived from PCA transformation.
- **Amount**: Transaction amount.
- **Class**: Target variable (1 for fraud, 0 for valid transaction).

## Requirements

To run this project, ensure you have the following Python libraries installed:
- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `imbalanced-learn`

Install them using:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn
