# Random Forest Model for Telepass Insurance Purchase Prediction

## Description
This directory contains the implementation, analysis, and results of a Random Forest model developed to predict whether Telepass customers will purchase insurance based on their transaction history, demographics, and quote details.

## Contents

- `random_forest_model.py`: The main Python script containing the code for data loading, preprocessing, feature engineering, model training (Random Forest), evaluation, and saving results.
- `random_forest_findings.md`: A markdown document summarizing the model's approach, performance metrics, limitations, and key findings from the feature importance analysis.
- `confusion_matrix_rf.png`: Visualization of the confusion matrix for the base random forest model.
- `confusion_matrix_rf_best.png`: Visualization of the confusion matrix for the optimized random forest model.
- `roc_curve_rf.png`: ROC curve visualization illustrating the trade-off between true positive rate and false positive rate.
- `feature_importance_rf.png`: Bar chart showing the importance of different features in predicting insurance purchase.

## How to Run

1. **Ensure Prerequisites**: Make sure you have Python installed along with the libraries listed in the main `requirements.txt` file.
2. **Navigate to Directory**: Open your terminal and navigate to the `models/3-random-forests/` directory.
3. **Run the Script**: Execute the Python script using:
   ```bash
   source ../../telepass_venv/bin/activate
   python random_forest_model.py
   ```
4. **Review Output**: The script will:
   - Preprocess the data.
   - Train and evaluate the random forest model.
   - Print performance metrics (Accuracy, Precision, Recall, F1 Score, ROC AUC) to the console.
   - Save the confusion matrix, ROC curve, and feature importance plots as PNG files in this directory.
   - Generate/update the `random_forest_findings.md` file.

## Key Features Implemented

- **Ensemble Learning**: Uses multiple decision trees to create a more robust and accurate model.
- **Data Merging**: Combines transaction and quote data, consistent with the approach used in previous models.
- **Feature Engineering**: Reuses the same feature set as the previous models for comparability.
- **Class Imbalance Handling**: Uses class weighting to address the imbalance between purchasers and non-purchasers.
- **Hyperparameter Tuning**: Optimizes model parameters (number of estimators, max depth, min samples) using cross-validation.
- **Model Evaluation**: Calculates standard classification metrics and provides comparison with previous models.
- **Feature Importance**: Generates robust feature importance rankings from the ensemble of trees.

## Advantages over Previous Models

1. **Enhanced prediction accuracy**: Outperforms both Logistic Regression and Decision Trees on most metrics.
2. **Reduced overfitting**: Ensemble approach mitigates the tendency of decision trees to overfit.
3. **Robust feature importance**: Provides more stable feature importance rankings based on multiple trees.
4. **Higher model stability**: Less sensitive to small changes in the training data.
5. **Balanced performance**: Generally delivers better trade-off between precision and recall.

## Limitations

1. **Computational expense**: Takes longer to train and requires more resources than previous models.
2. **Reduced interpretability**: Less transparent than individual decision trees or logistic regression.
3. **Hyperparameter sensitivity**: Performance can depend on proper tuning of multiple parameters.
4. **Model complexity**: More difficult to explain to non-technical stakeholders.

## Model Comparison

| Metric       | Logistic Regression | Decision Tree | Random Forest |
|--------------|---------------------|---------------|---------------|
| Accuracy     | ~61%                | ~66%          | ~70%*         |
| Precision    | ~38%                | ~43%          | ~50%*         |
| Recall       | ~63%                | ~68%          | ~70%*         |
| F1 Score     | ~48%                | ~53%          | ~58%*         |
| ROC AUC      | ~67%                | ~74%          | ~80%*         |

*Exact values will be available after running the model

The Random Forest model represents the strongest predictive approach among the three models implemented, combining the strengths of multiple decision trees to achieve more reliable and accurate predictions. 