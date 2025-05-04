# Decision Tree Model for Telepass Insurance Purchase Prediction

## Description
This directory contains the implementation, analysis, and results of a Decision Tree model developed to predict whether Telepass customers will purchase insurance based on their transaction history, demographics, and quote details.

## Contents

- `decision_tree_model.py`: The main Python script containing the code for data loading, preprocessing, feature engineering, model training (Decision Tree), evaluation, and saving results.
- `decision_tree_findings.md`: A markdown document summarizing the model's approach, performance metrics, limitations, and key findings from the feature importance analysis.
- `confusion_matrix_dt.png`: Visualization of the confusion matrix for the base decision tree model.
- `confusion_matrix_dt_best.png`: Visualization of the confusion matrix for the optimized decision tree model.
- `roc_curve_dt.png`: ROC curve visualization illustrating the trade-off between true positive rate and false positive rate.
- `feature_importance_dt.png`: Bar chart showing the importance of different features in predicting insurance purchase.
- `decision_tree_visualization.png`: Visual representation of the decision tree structure (limited to a few levels for clarity).

## How to Run

1. **Ensure Prerequisites**: Make sure you have Python installed along with the libraries listed in the main `requirements.txt` file.
2. **Navigate to Directory**: Open your terminal and navigate to the `models/2-decision-trees/` directory.
3. **Run the Script**: Execute the Python script using:
   ```bash
   python decision_tree_model.py
   ```
4. **Review Output**: The script will:
   - Preprocess the data.
   - Train and evaluate the decision tree model.
   - Print performance metrics (Accuracy, Precision, Recall, F1 Score, ROC AUC) to the console.
   - Save the confusion matrix, ROC curve, decision tree visualization, and feature importance plots as PNG files in this directory.
   - Generate/update the `decision_tree_findings.md` file.

## Key Features Implemented

- **Data Merging**: Combines transaction and quote data, consistent with the approach used in the logistic regression model.
- **Feature Engineering**: Reuses the same feature set as the logistic regression model for comparability.
- **Non-linear Pattern Capture**: Naturally identifies non-linear relationships and feature interactions.
- **Class Imbalance Handling**: Uses class weighting to address the imbalance between purchasers and non-purchasers.
- **Hyperparameter Tuning**: Optimizes model parameters (max depth, min samples, criterion) using cross-validation.
- **Tree Visualization**: Generates a visual representation of the decision tree structure.
- **Model Evaluation**: Calculates standard classification metrics and comparison with the logistic regression model.
- **Interpretable Rules**: Extracts clear decision rules that can be communicated to business stakeholders.

## Advantages over Logistic Regression

1. **Better handling of non-linear relationships** between features and the target
2. **Automatic feature selection** and handling of irrelevant features
3. **Natural capturing of feature interactions** without explicit engineering
4. **More robust to outliers and missing values**
5. **No distributional assumptions** about the data
6. **Visual interpretability** through decision tree diagrams

## Limitations

1. **Potential overfitting** to training data if not properly constrained
2. **Instability**: Small variations in the data can result in very different tree structures
3. **Axis-parallel splits**: Each decision boundary must be parallel to feature axes
4. **Bias toward high-cardinality features**: Features with more unique values tend to get higher importance 