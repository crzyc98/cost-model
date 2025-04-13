Objective
Develop and train a machine learning model to predict the probability of an employee terminating within the next year. This model will replace the rule-based turnover scoring currently used as the fallback in projection.py when use_ml_turnover is enabled.
1. Data Requirements
Source Data: Utilize multiple years of historical census data (e.g., the output of generate_census.py for several consecutive years). A minimum of two consecutive years is required to identify terminations, but more years are preferable for model robustness.
Target Variable Creation:
Process pairs of consecutive yearly census files (Year N and Year N+1).
For each employee present in the Year N census, create a binary target variable terminated_next_year.
terminated_next_year = 1 if the employee (identified by ssn or another unique ID) is not present in the Year N+1 census or has a termination_date recorded between the end of Year N and the end of Year N+1.
terminated_next_year = 0 if the employee is present in the Year N+1 census and does not have a termination date within that period.
Input Features: Extract or calculate relevant features from the Year N census data for predicting termination in Year N+1. Potential features include:
age: Calculated as of the end of Year N.
tenure: Calculated in years/months as of the end of Year N.
gross_compensation: Raw value from Year N.
role: Categorical feature (e.g., 'Staff', 'Manager', 'Executive').
pre_tax_deferral_percentage: Deferral rate in Year N.
is_participating: Boolean flag based on deferral rate in Year N.
(Optional) Compensation Rank/Percentile: Compensation relative to peers within the same role or overall.
(Optional) Recent Comp Increase: Difference between Year N and Year N-1 compensation (requires 3 years of data).
(Optional) Interaction terms (e.g., age * tenure).
2. Model Development Steps
Data Preparation & Preprocessing:
Load and combine historical census data.
Generate the terminated_next_year target variable as described above.
Calculate all necessary features based only on data available at the end of Year N.
Handle Missing Data: Investigate and apply appropriate strategies (e.g., imputation for numerical features if necessary, though ideally census data is complete).
Encode Categorical Features: Convert features like role into numerical representations using techniques like One-Hot Encoding (using sklearn.preprocessing.OneHotEncoder).
Feature Scaling: Apply standardization (sklearn.preprocessing.StandardScaler) or normalization (MinMaxScaler) to numerical features. This is crucial for many ML algorithms.
Data Splitting: Divide the prepared dataset into training and testing sets (e.g., 80% train, 20% test). Consider a time-based split if data covers many years (e.g., train on earlier years, test on the most recent year transition) to better simulate real-world prediction.
Feature Engineering (Optional but Recommended):
Explore creating new features from existing ones (e.g., interaction terms, compensation ratios/percentiles) that might improve predictive power. Document any engineered features.
Model Selection:
Frame the problem as binary classification (predicting terminated_next_year = 1 or 0).
Evaluate several suitable algorithms:
Baseline: Logistic Regression (sklearn.linear_model.LogisticRegression) - good for interpretability.
Ensemble Methods: Random Forest (sklearn.ensemble.RandomForestClassifier) or Gradient Boosting Machines (like XGBoost xgboost.XGBClassifier or LightGBM lightgbm.LGBMClassifier) - often provide higher accuracy.
Consider models that output probabilities (predict_proba) as the simulation uses the probability score for ranking.
Model Training & Evaluation:
Train selected models on the training dataset.
Address Class Imbalance: Terminations might be less frequent than non-terminations. Investigate if the dataset is imbalanced. If so, employ techniques like:
Class weighting (setting class_weight='balanced' in many Scikit-learn models).
Resampling techniques (e.g., SMOTE from the imbalanced-learn library).
Evaluate model performance on the testing dataset using appropriate metrics:
AUC-ROC Score (primary metric for probability models).
Precision, Recall, F1-Score (especially for the 'terminated' class).
Confusion Matrix.
Accuracy (less reliable if imbalanced).
Hyperparameter Tuning: Optimize the chosen model's hyperparameters using techniques like GridSearchCV or RandomizedSearchCV with cross-validation on the training set.
Final Model Packaging:
Combine the final preprocessing steps (encoding, scaling) and the trained model into a single Scikit-learn Pipeline (sklearn.pipeline.Pipeline). This ensures the same transformations are applied consistently during prediction.
Save the entire trained Pipeline object to a file using joblib or pickle.
Separately, save the exact list of feature column names (model_feature_names) that the pipeline expects as input, in the correct order.
3. Integration with projection.py
The saved Pipeline object (.joblib or .pkl file) and the list of feature names are the deliverables needed for the simulation script.
The projection.py script (specifically the project_census function) will:
Load the saved Pipeline object when use_ml_turnover is True and a valid ml_model_path is provided.
Use the existing prepare_features_for_model function to generate the raw features from the simulation's current_df.
Ensure the feature DataFrame passed to the model has columns matching the saved model_feature_names list, in the correct order.
Use the loaded pipeline's predict_proba method: pipeline.predict_proba(features)[:, 1] to get the probability of termination (class 1).
Use these probabilities as the turnover_score for ranking employees for termination, as already implemented in the existing logic.
4. Technical Considerations
Libraries: Primarily use Scikit-learn, Pandas, NumPy. Potentially XGBoost, LightGBM, imbalanced-learn. Use joblib or pickle for saving the model pipeline.
Reproducibility: Use version control (e.g., Git). Document data sources, preprocessing steps, feature engineering choices, model parameters, and evaluation results clearly. Set random seeds for reproducibility where applicable.
Feature Importance: Analyze feature importances (available from models like Logistic Regression, Random Forest, Gradient Boosting) to understand drivers of termination.
