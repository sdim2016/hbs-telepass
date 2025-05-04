# Logistic Regression Model for Telepass Insurance Purchase Prediction

## Description
This directory contains the implementation, analysis, and results of a Logistic Regression model developed to predict whether Telepass customers will purchase insurance based on their transaction history, demographics, and quote details.

## Contents

- `logistic_regression_model.py`: The main Python script containing the code for data loading, preprocessing, feature engineering, model training (Logistic Regression), evaluation, and saving results.
- `logistic_regression_model.ipynb`: A Jupyter notebook version of the analysis, providing a more interactive environment for exploration and visualization.
- `logistic_regression_findings.md`: A markdown document summarizing the model's approach, performance metrics, limitations, and key findings from the feature importance analysis.
- `confusion_matrix_logistic.png`: Visualization of the confusion matrix for the base logistic regression model.
- `confusion_matrix_logistic_best.png`: Visualization of the confusion matrix for the optimized logistic regression model.
- `roc_curve_logistic.png`: ROC curve visualization illustrating the trade-off between true positive rate and false positive rate.
- `feature_importance_logistic.png`: Bar chart showing the importance of different features in predicting insurance purchase.

## How to Run

1.  **Ensure Prerequisites**: Make sure you have Python installed along with the libraries listed in the main `requirements.txt` file.
2.  **Navigate to Directory**: Open your terminal and navigate to the `models/1-logistic-regression/` directory.
3.  **Run the Script**: Execute the Python script using:
    ```bash
    python logistic_regression_model.py
    ```
4.  **Review Output**: The script will:
    - Preprocess the data.
    - Train and evaluate the logistic regression model.
    - Print performance metrics (Accuracy, Precision, Recall, F1 Score, ROC AUC) to the console.
    - Save the confusion matrix, ROC curve, and feature importance plots as PNG files in this directory.
    - Generate/update the `logistic_regression_findings.md` file.
5.  **Explore the Notebook (Optional)**: Open and run the cells in `logistic_regression_model.ipynb` using Jupyter Lab or Jupyter Notebook for an interactive walkthrough.

## Key Features Implemented

- **Data Merging**: Combines transaction and quote data.
- **Feature Engineering**: Creates derived features like age, membership duration, and transaction patterns.
- **Data Preprocessing**: Handles missing values, standardizes numerical features, and encodes categorical features.
- **Class Imbalance Handling**: Uses class weighting to address the imbalance between purchasers and non-purchasers.
- **Hyperparameter Tuning**: Optimizes model parameters (regularization strength, penalty type) using cross-validation.
- **Model Evaluation**: Calculates standard classification metrics.
- **Visualization**: Generates plots for confusion matrix, ROC curve, and feature importance.
- **Findings Documentation**: Summarizes results and insights in a markdown file.

## Performance Summary

- **Accuracy**: ~61.3%
- **Recall**: ~63.0%
- **Precision**: ~38.3%
- **F1 Score**: ~47.6%
- **ROC AUC**: ~67.0%

*(Note: Exact values may vary slightly depending on the train/test split.)*

## Key Findings

- Transaction behavior (especially total transactions) is the most significant predictor.
- Telepass subscription type (Premium) and membership duration are influential.
- Broker ID plays a noticeable role in purchase likelihood.

Refer to `logistic_regression_findings.md` for a detailed analysis. 