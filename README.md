# Telepass Insurance Purchase Prediction Analysis

## Project Goal

This project aims to analyze Telepass customer data to predict the likelihood of purchasing an insurance policy offered via a quote. We will explore customer transaction patterns, demographics, and car information to build predictive models.

## Project Structure

```
/
├── data/
│   ├── transactions.csv
│   ├── insurance_quotes.csv
│   └── data_dictionary_*.csv
├── models/
│   ├── 1-logistic-regression/
│   │   ├── logistic_regression_model.py
│   │   ├── logistic_regression_model.ipynb
│   │   ├── logistic_regression_findings.md
│   │   ├── README.md
│   │   └── *.png # Visualization outputs
│   ├── 2-decision-trees/
│   │   ├── decision_tree_model.py
│   │   ├── decision_tree_model.ipynb
│   │   ├── decision_tree_findings.md
│   │   ├── README.md
│   │   └── *.png # Visualization outputs
│   └── 3-random-forests/
│       ├── random_forest_model.py
│       ├── random_forest_model.ipynb
│       ├── random_forest_findings.md
│       ├── README.md
│       └── *.png # Visualization outputs
├── docs/
│   ├── Telepass.md
│   └── assessing-prediction-accuracy-of-machine-learning-models.md
├── telepass_analysis.py # Initial analysis script (might be refactored/removed)
├── requirements.txt
└── README.md
└── SUMMARY.md
```

## Data

- `transactions.csv`: Contains customer transaction history with Telepass services.
- `insurance_quotes.csv`: Contains details about insurance quotes offered to customers and whether they purchased.
- `data_dictionary_*.csv`: Provides descriptions for the columns in the datasets.

*(Located in the `data/` directory)*

## Models

1.  **Logistic Regression (`models/1-logistic-regression/`)**: 
    - A baseline model providing interpretable insights.
    - Includes Python script (`.py`), Jupyter Notebook (`.ipynb`), findings (`.md`), and visualizations (`.png`).
2.  **Decision Trees (`models/2-decision-trees/`)**: 
    - Implemented model capturing non-linear relationships and feature interactions.
    - Includes Python script (`.py`), Jupyter Notebook (`.ipynb`), findings (`.md`), and visualizations (`.png`).
3.  **Random Forests (`models/3-random-forests/`)**: 
    - Ensemble model leveraging multiple decision trees for improved performance.
    - Includes Python script (`.py`), Jupyter Notebook (`.ipynb`), findings (`.md`), and visualizations (`.png`).

## Setup and Installation

1.  **Clone the repository (if applicable)**
2.  **Create a virtual environment**: 
    ```bash
    python -m venv telepass_venv
    source telepass_venv/bin/activate # On Windows use `telepass_venv\Scripts\activate`
    ```
3.  **Install dependencies**: 
    ```bash
    pip install -r requirements.txt
    ```

## How to Run Models

### Using Python Scripts

Navigate to the specific model directory (e.g., `models/1-logistic-regression/`) and follow the instructions in its `README.md` file. Generally, you will run the Python script associated with the model after activating the virtual environment:

```bash
# Activate environment
source telepass_venv/bin/activate

# Navigate to model directory
cd models/1-logistic-regression

# Run the model
python logistic_regression_model.py
```

### Using Jupyter Notebooks

For an interactive exploration of the models:

```bash
# Activate environment
source telepass_venv/bin/activate

# Start Jupyter
jupyter lab
```

Then navigate to the relevant model directory and open the corresponding notebook:
- `models/1-logistic-regression/logistic_regression_model.ipynb`
- `models/2-decision-trees/decision_tree_model.ipynb`
- `models/3-random-forests/random_forest_model.ipynb`

## Requirements

The project requires Python 3.6+ and the following packages:
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- jupyterlab/notebook (for interactive execution)

## Model Implementations

### 1. Logistic Regression

The Logistic Regression model uses customer profile data and transaction history to predict insurance purchase likelihood. Details can be found in [models/1-logistic-regression/README.md](models/1-logistic-regression/README.md).

#### Performance
- Accuracy: ~61%
- Precision: ~38%
- Recall: ~63%
- F1 Score: ~48%
- ROC AUC: ~67%

### 2. Decision Trees

The Decision Tree model builds on the insights from the Logistic Regression model, seeking to capture non-linear relationships and feature interactions. Details can be found in [models/2-decision-trees/README.md](models/2-decision-trees/README.md).

#### Performance
- Accuracy: ~66%
- Precision: ~43%
- Recall: ~68%
- F1 Score: ~53%
- ROC AUC: ~74%

### 3. Random Forests

The Random Forest model leverages ensemble methods to improve prediction accuracy beyond the capabilities of individual decision trees. It combines multiple decision trees to create more robust predictions. Details can be found in [models/3-random-forests/README.md](models/3-random-forests/README.md).

#### Performance
- Accuracy: ~73%
- Precision: ~51%
- Recall: ~46%
- F1 Score: ~49%
- ROC AUC: ~73%

## Model Comparison and Findings

### Performance Comparison

| Metric       | Logistic Regression | Decision Tree | Random Forest |
|--------------|---------------------|---------------|---------------|
| Accuracy     | 61.3%               | 65.7%         | 72.8%         |
| Precision    | 38.3%               | 42.8%         | 51.5%         |
| Recall       | 63.0%               | 67.9%         | 46.0%         |
| F1 Score     | 47.6%               | 52.5%         | 48.6%         |
| ROC AUC      | 67.0%               | 74.5%         | 73.2%         |

### Key Findings

1. **Transaction patterns are crucial**: All models identified transaction behavior as a key predictor, suggesting that customer engagement is strongly linked to purchase likelihood.

2. **Non-linear relationships matter**: The significant improvement from Logistic Regression to Decision Tree models indicates the presence of important non-linear relationships in the data.

3. **Price sensitivity varies**: The Random Forest model highlighted pricing factors as more influential than the other models, suggesting price optimization could impact sales.

4. **Membership characteristics influence decisions**: Telepass membership duration and subscription types consistently appeared as important features across models.

5. **Model selection depends on business goals**:
   - For maximizing customer identification (recall): Decision Trees perform best
   - For precision and accuracy: Random Forests provide the strongest results
   - For interpretability: Logistic Regression offers the clearest insights