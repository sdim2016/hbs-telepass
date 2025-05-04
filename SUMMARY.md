# Telepass Insurance Purchase Prediction Analysis - Summary

This document summarizes the modeling approach, performance, optimization choices, and comparative analysis for the three predictive models implemented in this project.

## Part 1: Prediction Models

### 1. Model Building Approach

#### Logistic Regression Model
- **Modeling Approach**: We started with Logistic Regression as a baseline due to its interpretability and efficiency. We merged transaction data with insurance quotes, preprocessed the data (handling missing values, date conversions), and applied feature engineering.
- **Modeling Decisions**: Guided by the need for a transparent model that provides clear coefficient interpretation. We implemented class balancing to handle the imbalanced dataset (72% non-purchases vs 28% purchases) and used standardization for numerical features and one-hot encoding for categorical features.
- **Variables Included**: 
  - **Numerical**: Basic coverage, price information, transaction metrics, car age, membership duration
  - **Categorical**: Driving type, car brand, gender, operating system, broker ID, subscription status
  - **Derived Features**: Customer age, car age, Telepass membership duration, flags for active subscriptions, transaction patterns

#### Decision Tree Model
- **Modeling Approach**: Building on the Logistic Regression model, we implemented Decision Trees to capture non-linear relationships and feature interactions that linear models might miss. We maintained the same data preparation and feature engineering process for consistency.
- **Modeling Decisions**: Guided by the desire to capture complex relationships and decision boundaries. We implemented cross-validation and hyperparameter tuning to find the optimal tree structure, focusing on controlling depth to prevent overfitting.
- **Variables Included**: Same feature set as the Logistic Regression model for direct comparison, allowing us to evaluate the impact of the model architecture rather than feature differences.

#### Random Forest Model
- **Modeling Approach**: We extended our analysis with a Random Forest model to leverage ensemble learning, combining multiple decision trees to improve prediction stability and accuracy. The data preparation and feature engineering remained consistent with previous models.
- **Modeling Decisions**: Guided by the goal of addressing decision tree weaknesses (overfitting, instability) while maintaining their ability to capture non-linear patterns. We optimized the forest structure through hyperparameter tuning of estimator count, tree depth, and minimum samples.
- **Variables Included**: Same feature set as previous models, ensuring comparability across all three approaches.

### 2. Model Performance and Limitations

#### Logistic Regression Performance
- **Accuracy**: 61.27%
- **Precision**: 38.29%
- **Recall**: 62.98%
- **F1 Score**: 47.63%
- **ROC AUC**: 66.97%
- **Strengths**: High recall (identifies many potential buyers), excellent interpretability, computational efficiency
- **Limitations**: 
  - Limited precision (high false positive rate)
  - Linearity assumption restricts ability to capture complex relationships
  - Struggles with the imbalanced dataset despite weighting
  - Cannot naturally capture feature interactions

#### Decision Tree Performance
- **Accuracy**: 65.69%
- **Precision**: 42.84%
- **Recall**: 67.87%
- **F1 Score**: 52.52%
- **ROC AUC**: 74.46%
- **Strengths**: Captures non-linear relationships, automatically identifies important features, handles interactions naturally, transparent decision rules
- **Limitations**:
  - Potential overfitting despite pruning efforts
  - Instability (small data variations can create different trees)
  - Limited boundary complexity (only parallel to feature axes)
  - Bias toward high-cardinality features

#### Random Forest Performance
- **Accuracy**: 72.77%
- **Precision**: 51.46%
- **Recall**: 46.02%
- **F1 Score**: 48.59%
- **ROC AUC**: 73.21%
- **Strengths**: Ensemble power reduces overfitting, provides robust feature importance, captures complex patterns, reduced sensitivity to noise
- **Limitations**:
  - Computationally expensive (training and prediction)
  - Reduced interpretability compared to single trees or logistic regression
  - Longer training time
  - Higher memory requirements

### 3. Optimization Metrics and Choices

#### Common Optimization Approach
Across all three models, we maintained a consistent optimization approach:

- **Primary Metric - ROC AUC**: We chose this metric to guide hyperparameter tuning because:
  - It is insensitive to class imbalance (important given our 72%/28% split)
  - It evaluates model performance across all classification thresholds
  - It measures the model's ability to rank positive instances higher than negative ones

- **Secondary Metric - Recall**: We emphasized recall because, from a business perspective, the cost of missing a potential customer (false negative) is likely higher than incorrectly targeting a non-buyer (false positive).

- **Class Balancing**: All models implemented class weighting to address the imbalance, giving higher weight to the minority class (insurance purchasers).

#### Model-Specific Optimization

- **Logistic Regression**:
  - Optimized regularization strength (C) and penalty type
  - Best performance with C=0.1 and L1 regularization
  
- **Decision Tree**:
  - Focused on controlling complexity through max depth, min samples split/leaf
  - Tested both Gini impurity and entropy as split criteria
  - Emphasized pre-pruning over post-pruning

- **Random Forest**:
  - Tuned number of estimators, max depth, and minimum samples
  - Focused on finding the balance between model complexity and generalization

### 4. Best Performing Model

Based on our analysis, the **best model depends on the specific business objective**:

#### Performance Comparison

| Metric       | Logistic Regression | Decision Tree | Random Forest |
|--------------|---------------------|---------------|---------------|
| Accuracy     | 61.3%               | 65.7%         | 72.8%         |
| Precision    | 38.3%               | 42.8%         | 51.5%         |
| Recall       | 63.0%               | 67.9%         | 46.0%         |
| F1 Score     | 47.6%               | 52.5%         | 48.6%         |
| ROC AUC      | 67.0%               | 74.5%         | 73.2%         |

#### Best Model by Objective:

1. **If maximizing accuracy and precision is the goal** (identifying true buyers with minimal false positives), the **Random Forest model** performed best with 72.8% accuracy and 51.5% precision.

2. **If maximizing recall is the priority** (identifying as many potential buyers as possible, even at the cost of some false positives), the **Decision Tree model** is optimal with 67.9% recall.

3. **If balanced performance (F1 Score) is desired**, the **Decision Tree model** provides the best balance with an F1 score of 52.5%.

4. **If ROC AUC is the metric of choice** (overall ranking capability), the **Decision Tree model** slightly outperforms with 74.5%.

5. **If interpretability is crucial**, the **Logistic Regression** or **Decision Tree** models offer clearer insights into the factors driving predictions.

#### Why These Performance Differences?

- **Random Forest excels in accuracy/precision**: The ensemble approach reduces overfitting and creates more stable decision boundaries, leading to fewer false positives.

- **Decision Tree leads in recall**: Its ability to create complex decision boundaries allows it to capture more potential buyers, though at the cost of more false positives.

- **Different feature importance rankings**: While transaction behavior is important across all models, pricing factors gain more prominence in the Random Forest model, suggesting they interact in complex ways that the ensemble method captures more effectively.

## Part 2: Case Reflection

### Insurance Strategy Recommendation for Telepass

Based on our analysis of both the Telepass case and our predictive modeling results, we recommend that Cervellin should continue with the insurance brokerage model in the short to medium term, while strategically positioning for a potential direct insurance offering in the future. This phased approach balances opportunity with prudent risk management.

Telepass has built a valuable position as an intermediary in the insurance space since launching Telepass Broker in June 2019. In just eight months, they facilitated over 10,000 insurance policy sales. The current brokerage model provides several advantages: it leverages Telepass's unique customer data without requiring the substantial regulatory overhead of becoming a direct insurer, it avoids direct competition with current partners, and it generates revenue through commissions on converted leads with minimal risk exposure.

Our predictive modeling work has demonstrated that Telepass's data provides significant value in identifying likely insurance purchasers. The transaction patterns we identified as strong predictors align with Telepass's unique visibility into customer mobility behaviors. This data advantage is real and substantive – Telepass can see that while two 40-year-old men living in Milan might appear identical to traditional insurers, one might commute daily while the other rarely drives, representing fundamentally different risk profiles. However, our models also showed limitations in precision (51.5% at best with Random Forest), indicating that predicting insurance purchases remains challenging even with rich data.

The key question is whether this data advantage is sufficient to justify the significant leap to becoming a direct insurer. Moving to direct insurance sales would require creating another separate company to comply with regulations, developing underwriting expertise, establishing reinsurance relationships, and managing capital reserves – all substantial investments. This would also position Telepass in direct competition with current Telepass Broker merchants, potentially damaging valuable partnerships. Furthermore, the current economic uncertainty due to COVID-19 makes this a particularly risky time for major strategic pivots in the mobility sector.

A more prudent approach would be to further develop the brokerage model while building the capabilities that could enable a direct insurance offering in the future. Cervellin should focus on three priorities: first, enhance the predictive models for the current brokerage offering – our analysis indicates that the Random Forest model could be particularly valuable for precision targeting of high-probability customers; second, develop a deeper understanding of risk pricing through analysis of insurance purchase data across different customer segments; and third, experiment with creating custom insurance "bundles" that could be offered through partners initially but might form the basis of direct products later.

This approach allows Telepass to leverage its current strengths – the TelepassPay platform, rich customer data, and established industry relationships – while gradually building the expertise needed for direct insurance offerings. The company has already demonstrated success with a gradual approach to innovation in its evolution from a tolling service to a mobility platform. The insurance market represents a similar opportunity for thoughtful expansion that builds on Telepass's core strength as an ecosystem integrator.

By continuing to refine its data advantage and understanding of customer insurance needs, Telepass can create more value in the short term through an enhanced brokerage offering while keeping the option open for direct insurance products when market conditions and organizational capabilities align. This strategy respects the uncertainty of the current environment while positioning the company for future growth in the insurance vertical.

## Conclusion

Each model brings distinct strengths to the prediction task:

- **Logistic Regression**: Provides a solid baseline with interpretable coefficients
- **Decision Tree**: Effectively captures non-linear relationships with transparent decision rules
- **Random Forest**: Delivers the highest accuracy and precision through ensemble learning

The optimal model choice depends on the business context and specific goals of the prediction task. If the business prioritizes identifying as many potential customers as possible, the Decision Tree model would be preferred. If minimizing false positives and achieving higher overall accuracy is the goal, the Random Forest model would be the better choice. 