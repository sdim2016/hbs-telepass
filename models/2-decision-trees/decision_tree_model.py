#!/usr/bin/env python
# coding: utf-8

# # Telepass Insurance Prediction Models
# ## Model 2: Decision Trees

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve
import warnings
warnings.filterwarnings('ignore')

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)

# Load the datasets
print("Loading datasets...")
insurance_quotes = pd.read_csv('../../data/insurance_quotes.csv', sep=';')
transactions = pd.read_csv('../../data/transactions.csv', sep=';')

# Display basic information about the datasets
print("\nInsurance Quotes Dataset:")
print("Shape: {}".format(insurance_quotes.shape))
print(insurance_quotes.head())

print("\nTransactions Dataset:")
print("Shape: {}".format(transactions.shape))
print(transactions.head())

# Check for missing values
print("\nMissing values in Insurance Quotes Dataset:")
print(insurance_quotes.isnull().sum())

print("\nMissing values in Transactions Dataset:")
print(transactions.isnull().sum())

# Define a function to preprocess the insurance quotes dataset
def preprocess_insurance_quotes(df):
    """
    Preprocess the insurance quotes dataset for modeling.
    """
    # Make a copy of the dataframe
    df_processed = df.copy()
    
    # Convert categorical variables to appropriate types
    
    # Handle date columns
    date_columns = ['car_immatriculation_date', 'insurance_expires_at', 'birth_date', 
                   'base_subscription', 'pay_subscription', 'pay_cancellation',
                   'premium_subscription', 'premium_cancellation', 'policy_quoted_at']
    
    for col in date_columns:
        if col in df_processed.columns:
            try:
                df_processed[col] = pd.to_datetime(df_processed[col], errors='coerce')
            except:
                print("Could not convert {} to datetime.".format(col))
    
    # Handle the target variable - properly convert 'TRUE'/'FALSE' to 1/0
    if 'issued' in df_processed.columns:
        df_processed['issued'] = df_processed['issued'].map({True: 1, 'TRUE': 1, False: 0, 'FALSE': 0})
    
    # Convert numeric columns with comma separators
    numeric_columns = ['driver_injury', 'basic_coverage', 'legal_protection', 
                       'waive_right_compensation', 'uninsured_vehicles', 
                       'protected_bonus', 'windows', 'natural_events', 
                       'theft_fire', 'kasko', 'license_revoked', 
                       'collision', 'vandalism', 'key_loss', 
                       'price_sale', 'price_full', 'discount_percent']
    
    for col in numeric_columns:
        if col in df_processed.columns:
            df_processed[col] = df_processed[col].astype(str).str.replace(',', '.').astype(float, errors='ignore')
    
    # Create derived features
    # Calculate age at policy quote time
    if 'birth_date' in df_processed.columns and 'policy_quoted_at' in df_processed.columns:
        mask = ~df_processed['birth_date'].isna() & ~df_processed['policy_quoted_at'].isna()
        df_processed.loc[mask, 'age'] = (df_processed.loc[mask, 'policy_quoted_at'] - 
                                         df_processed.loc[mask, 'birth_date']).dt.days / 365.25
    
    # Calculate car age at policy quote time
    if 'car_immatriculation_date' in df_processed.columns and 'policy_quoted_at' in df_processed.columns:
        mask = ~df_processed['car_immatriculation_date'].isna() & ~df_processed['policy_quoted_at'].isna()
        df_processed.loc[mask, 'car_age'] = (df_processed.loc[mask, 'policy_quoted_at'] - 
                                            df_processed['car_immatriculation_date']).dt.days / 365.25
    
    # Telepass membership duration
    if 'base_subscription' in df_processed.columns and 'policy_quoted_at' in df_processed.columns:
        mask = ~df_processed['base_subscription'].isna() & ~df_processed['policy_quoted_at'].isna()
        df_processed.loc[mask, 'telepass_membership_years'] = (df_processed.loc[mask, 'policy_quoted_at'] - 
                                                             df_processed['base_subscription']).dt.days / 365.25
    
    # Flag customers with TelepassPay
    if 'pay_subscription' in df_processed.columns and 'pay_cancellation' in df_processed.columns:
        df_processed['has_telepass_pay'] = 0
        # If they have a pay_subscription but no cancellation or cancellation is far in the future
        mask = (~df_processed['pay_subscription'].isna() & 
                (df_processed['pay_cancellation'].isna() | 
                 (df_processed['pay_cancellation'] > pd.to_datetime('2025-01-01'))))
        df_processed.loc[mask, 'has_telepass_pay'] = 1
    
    # Flag customers with Telepass Premium
    if 'premium_subscription' in df_processed.columns and 'premium_cancellation' in df_processed.columns:
        df_processed['has_telepass_premium'] = 0
        # If they have a premium_subscription but no cancellation or cancellation is far in the future
        mask = (~df_processed['premium_subscription'].isna() & 
                (df_processed['premium_cancellation'].isna() | 
                 (df_processed['premium_cancellation'] > pd.to_datetime('2025-01-01'))))
        df_processed.loc[mask, 'has_telepass_premium'] = 1
    
    # Count available guarantees
    if 'guarantees_available' in df_processed.columns:
        df_processed['num_guarantees_available'] = df_processed['guarantees_available'].str.count('-') + 1
    
    return df_processed

# Define a function to preprocess the transactions dataset
def preprocess_transactions(df):
    """
    Preprocess the transactions dataset for modeling.
    """
    # Make a copy of the dataframe
    df_processed = df.copy()
    
    # Convert year_month to datetime
    if 'year_month' in df_processed.columns:
        try:
            df_processed['year_month'] = pd.to_datetime(df_processed['year_month'], format='%b.%y', errors='coerce')
        except:
            print("Could not convert year_month to datetime.")
    
    # Handle numerical values
    if 'expenditures' in df_processed.columns:
        df_processed['expenditures'] = df_processed['expenditures'].astype(str).str.replace(',', '.').astype(float)
    
    return df_processed

# Function to create aggregate features per client from transactions
def create_transaction_features(df):
    """
    Aggregate transaction data to create features at the client level.
    """
    # Calculate total transactions and expenditures per client
    client_stats = df.groupby('client_id').agg(
        total_transactions=('number_transactions', 'sum'),
        total_expenditures=('expenditures', 'sum'),
        avg_expenditure_per_transaction=('expenditures', lambda x: x.sum() / df.loc[x.index, 'number_transactions'].sum()),
        num_months_active=('year_month', 'nunique'),
        num_service_types=('service_type', 'nunique')
    )
    
    # Calculate statistics for telepass_pay transactions
    telepass_pay_stats = df[df['telepass_pay'] == 1].groupby('client_id').agg(
        telepass_pay_transactions=('number_transactions', 'sum'),
        telepass_pay_expenditures=('expenditures', 'sum')
    )
    
    # Merge all features
    client_features = client_stats.copy()
    client_features = client_features.join(telepass_pay_stats, how='left')
    
    # Fill NaN values for clients without telepass_pay transactions
    client_features['telepass_pay_transactions'] = client_features['telepass_pay_transactions'].fillna(0)
    client_features['telepass_pay_expenditures'] = client_features['telepass_pay_expenditures'].fillna(0)
    
    # Calculate percentage of telepass_pay transactions
    client_features['telepass_pay_pct'] = client_features['telepass_pay_transactions'] / client_features['total_transactions']
    client_features['telepass_pay_expenditures_pct'] = client_features['telepass_pay_expenditures'] / client_features['total_expenditures']
    
    # Handle division by zero
    client_features.replace([np.inf, -np.inf], np.nan, inplace=True)
    client_features.fillna(0, inplace=True)
    
    # Reset index to make client_id a column
    client_features.reset_index(inplace=True)
    
    return client_features

# Preprocess the datasets
print("\nPreprocessing datasets...")
insurance_quotes_processed = preprocess_insurance_quotes(insurance_quotes)
transactions_processed = preprocess_transactions(transactions)

# Display the preprocessed datasets
print("\nPreprocessed Insurance Quotes Dataset:")
print(insurance_quotes_processed.head())

print("\nPreprocessed Transactions Dataset:")
print(transactions_processed.head())

# Create transaction features
print("\nCreating transaction features...")
transaction_features = create_transaction_features(transactions_processed)
print(transaction_features.head())

# Merge the datasets
print("\nMerging datasets...")
merged_data = insurance_quotes_processed.merge(
    transaction_features, 
    on='client_id', 
    how='left'
)

# Fill missing transaction data for clients without transactions
for col in transaction_features.columns:
    if col != 'client_id':
        merged_data[col] = merged_data[col].fillna(0)

print("Shape of merged data: {}".format(merged_data.shape))
print(merged_data.head())

# Check the values in the issued column
print("\nValues in the issued column:")
print(merged_data['issued'].value_counts(dropna=False))

# Drop rows with NaN in the target variable
print("\nRemoving rows with missing target values...")
merged_data = merged_data.dropna(subset=['issued'])
print("Shape after removing rows with missing target: {}".format(merged_data.shape))

# Feature selection for the model
print("\nSelecting features for modeling...")

# Numerical features
numerical_features = [
    'basic_coverage', 'price_sale', 'price_full', 'discount_percent',
    'car_age', 'telepass_membership_years', 'num_guarantees_available',
    'total_transactions', 'total_expenditures', 'avg_expenditure_per_transaction',
    'num_months_active', 'num_service_types', 'telepass_pay_transactions',
    'telepass_pay_expenditures', 'telepass_pay_pct', 'telepass_pay_expenditures_pct'
]

# Categorical features
categorical_features = [
    'driving_type', 'car_brand', 'gender', 'operating_system', 'broker_id',
    'roadside_assistance', 'has_telepass_pay', 'has_telepass_premium', 'base_type'
]

# Target variable
target = 'issued'

# Filter to only keep the features we want
selected_features = numerical_features + categorical_features
model_data = merged_data[selected_features + [target]].copy()

# Handle missing values
model_data = model_data.fillna({
    col: model_data[col].median() for col in numerical_features if col in model_data.columns
})

# Fill categorical features with mode
for col in categorical_features:
    if col in model_data.columns:
        model_data[col] = model_data[col].fillna(model_data[col].mode()[0])

# Print some basic statistics about the final dataset
print("\nFinal dataset shape: {}".format(model_data.shape))
print("\nClass distribution:")
print(model_data[target].value_counts(normalize=True))

# Split the data into train and test sets
X = model_data.drop(target, axis=1)
y = model_data[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\nTraining set shape: {}".format(X_train.shape))
print("Test set shape: {}".format(X_test.shape))

# Define preprocessing for numerical and categorical features
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, [col for col in numerical_features if col in X_train.columns]),
        ('cat', categorical_transformer, [col for col in categorical_features if col in X_train.columns])
    ]
)

# Create a pipeline with preprocessing and model
print("\nBuilding Decision Tree model...")
dt_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', DecisionTreeClassifier(random_state=42, class_weight='balanced'))
])

# Train the model
dt_pipeline.fit(X_train, y_train)

# Make predictions
y_pred = dt_pipeline.predict(X_test)
y_prob = dt_pipeline.predict_proba(X_test)[:, 1]

# Evaluate the model
print("\nModel Evaluation:")
print("Accuracy: {:.4f}".format(accuracy_score(y_test, y_pred)))
print("Precision: {:.4f}".format(precision_score(y_test, y_pred)))
print("Recall: {:.4f}".format(recall_score(y_test, y_pred)))
print("F1 Score: {:.4f}".format(f1_score(y_test, y_pred)))
print("ROC AUC: {:.4f}".format(roc_auc_score(y_test, y_prob)))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Plot confusion matrix
plt.figure(figsize=(8, 6))
conf_matrix = confusion_matrix(y_test, y_pred, normalize='true')
sns.heatmap(conf_matrix, annot=True, fmt='.2%', cmap='Blues')
plt.title('Normalized Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.savefig('confusion_matrix_dt.png')
plt.close()

# Plot ROC curve
plt.figure(figsize=(8, 6))
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.plot(fpr, tpr, label='ROC curve (area = {:.4f})'.format(roc_auc_score(y_test, y_prob)))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.savefig('roc_curve_dt.png')
plt.close()

# Hyperparameter tuning
print("\nTuning Hyperparameters...")
param_grid = {
    'classifier__max_depth': [None, 5, 10, 15, 20],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4, 8],
    'classifier__criterion': ['gini', 'entropy']
}

grid_search = GridSearchCV(
    dt_pipeline,
    param_grid,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print("Best Parameters: {}".format(grid_search.best_params_))
print("Best ROC AUC Score: {:.4f}".format(grid_search.best_score_))

# Evaluate the best model
best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test)
y_prob_best = best_model.predict_proba(X_test)[:, 1]

print("\nBest Model Evaluation:")
print("Accuracy: {:.4f}".format(accuracy_score(y_test, y_pred_best)))
print("Precision: {:.4f}".format(precision_score(y_test, y_pred_best)))
print("Recall: {:.4f}".format(recall_score(y_test, y_pred_best)))
print("F1 Score: {:.4f}".format(f1_score(y_test, y_pred_best)))
print("ROC AUC: {:.4f}".format(roc_auc_score(y_test, y_prob_best)))

print("\nClassification Report for Best Model:")
print(classification_report(y_test, y_pred_best))

# Plot confusion matrix for best model
plt.figure(figsize=(8, 6))
conf_matrix_best = confusion_matrix(y_test, y_pred_best, normalize='true')
sns.heatmap(conf_matrix_best, annot=True, fmt='.2%', cmap='Blues')
plt.title('Normalized Confusion Matrix (Best Model)')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.savefig('confusion_matrix_dt_best.png')
plt.close()

# Plot the decision tree (first few levels only, for visualization purposes)
plt.figure(figsize=(20, 10))
max_depth_to_plot = 3  # Limit tree depth for visualization
decision_tree = best_model.named_steps['classifier']

# Get feature names after preprocessing
feature_names = []
numerical_feature_names = [col for col in numerical_features if col in X_train.columns]
feature_names.extend(numerical_feature_names)
categorical_feature_names = []
for col in [c for c in categorical_features if c in X_train.columns]:
    unique_values = X_train[col].unique()
    for val in unique_values:
        categorical_feature_names.append("{}_{}" .format(col, val))
feature_names.extend(categorical_feature_names)

# Plot the tree
# Truncate feature_names list if needed to match the actual number of features after preprocessing
n_features = best_model.named_steps['preprocessor'].transform(X_train.iloc[:1]).shape[1]
feature_names_truncated = feature_names[:n_features] if len(feature_names) > n_features else feature_names

try:
    plot_tree(
        decision_tree,
        feature_names=feature_names_truncated,
        class_names=['Not Purchased', 'Purchased'],
        filled=True,
        rounded=True,
        max_depth=max_depth_to_plot
    )
    plt.title('Decision Tree Visualization (Limited to Depth {})'.format(max_depth_to_plot))
    plt.savefig('decision_tree_visualization.png', dpi=300, bbox_inches='tight')
    plt.close()
except Exception as e:
    print("Error creating decision tree visualization:", e)

# Feature importance from the decision tree
if hasattr(best_model.named_steps['classifier'], 'feature_importances_'):
    # Get feature names after preprocessing (reusing from above)

    # Get feature importances
    importances = best_model.named_steps['classifier'].feature_importances_
    
    # Create a DataFrame to store feature importances
    try:
        feature_importance = pd.DataFrame({
            'Feature': feature_names_truncated,
            'Importance': importances
        })
        
        # Sort by importance
        feature_importance = feature_importance.sort_values('Importance', ascending=False).reset_index(drop=True)
        
        # Display top 20 features
        print("\nTop 20 Important Features:")
        print(feature_importance.head(20))
        
        # Plot feature importance
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Importance', y='Feature', data=feature_importance.head(20))
        plt.title('Top 20 Feature Importances')
        plt.tight_layout()
        plt.savefig('feature_importance_dt.png')
        plt.close()
    except Exception as e:
        print("Error creating feature importance plot:", e)

print("\nDecision Tree Model Analysis Completed.")
print("\nModel Strengths:")
print("1. Non-linear Relationships: Decision Trees can capture non-linear patterns in the data.")
print("2. Feature Interactions: Automatically captures interactions between features.")
print("3. Interpretability: Tree structure provides visual representation of decision rules.")
print("4. Feature Importance: Clear ranking of feature importance.")
print("5. No Assumptions: Does not make assumptions about the distribution of the data.")

print("\nModel Weaknesses:")
print("1. Overfitting: Decision Trees can easily overfit to training data if not properly constrained.")
print("2. Instability: Small changes in data can lead to completely different tree structures.")
print("3. Bias: Can be biased toward features with more levels in categorical variables.")
print("4. Limited Expressiveness: Single trees may not capture very complex patterns.")