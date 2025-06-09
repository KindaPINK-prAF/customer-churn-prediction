import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_selection import SelectKBest, f_classif
import xgboost as xgb
import warnings
from google.colab import files
warnings.filterwarnings('ignore')

# Load the dataset
print("Please upload 'customer_churn_data.csv' if not already uploaded.")
uploaded = files.upload()

try:
    data = pd.read_csv('/content/customer_churn_data.csv')
    print("File loaded successfully!")
except FileNotFoundError:
    print("Error: 'customer_churn_data.csv' not found. Please ensure the file was uploaded correctly.")
    raise

# Data diagnostics
print("Dataset Shape:", data.shape)
print("Class Distribution:\n", data['Churn'].value_counts(normalize=True))
print("Missing Values:\n", data.isnull().sum())
print("Duplicates:", data.duplicated().sum())
if data.duplicated().sum() > 0:
    data = data.drop_duplicates()
    print("Removed duplicates. New shape:", data.shape)

# Drop customerID
if 'customerID' in data.columns:
    data = data.drop('customerID', axis=1)

# Target and features
target_column = 'Churn'
if target_column not in data.columns:
    print(f"Target column '{target_column}' not found. Using last column as target.")
    target_column = data.columns[-1]

X = data.drop(target_column, axis=1)
y = data[target_column]

# Encode target variable
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
print(f"Encoded target classes: {label_encoder.classes_} -> {np.unique(y)}")

# Class weights
class_counts = pd.Series(y).value_counts()
scale_pos_weight = class_counts[0] / class_counts[1] if class_counts[1] > 0 else 1
print(f"Scale pos weight (No/Yes ratio): {scale_pos_weight:.2f}")

# Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=['object']).columns
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
print(f"Numerical columns: {numerical_cols.tolist()}")
print(f"Categorical columns: {categorical_cols.tolist()}")

# Target encoding for PaymentMethod
if 'PaymentMethod' in X.columns:
    payment_mean = X.join(pd.Series(y, name='Churn')).groupby('PaymentMethod')['Churn'].mean()
    X['PaymentMethod'] = X['PaymentMethod'].map(payment_mean)
    categorical_cols = categorical_cols.drop('PaymentMethod')
    numerical_cols = numerical_cols.append(pd.Index(['PaymentMethod']))
    print("Applied target encoding to PaymentMethod.")

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_cols)
    ])

# Define models
rf = RandomForestClassifier(random_state=42, class_weight='balanced', max_depth=5)
xgb_model = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss', scale_pos_weight=scale_pos_weight)

# Create pipelines with feature selection
rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('selector', SelectKBest(f_classif, k=10)),
    ('classifier', rf)
])
xgb_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('selector', SelectKBest(f_classif, k=10)),
    ('classifier', xgb_model)
])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Hyperparameter tuning for XGBoost
param_grid = {
    'classifier__n_estimators': [100, 200, 300, 500],
    'classifier__max_depth': [3, 5, 7, 9],
    'classifier__learning_rate': [0.01, 0.05, 0.1, 0.2],
    'classifier__subsample': [0.7, 0.9]
}
grid_search = GridSearchCV(xgb_pipeline, param_grid, cv=5, scoring='f1', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best model
best_model = grid_search.best_estimator_

# Evaluate models
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, average='binary', zero_division=0),
        'Recall': recall_score(y_test, y_pred, average='binary'),
        'F1 Score': f1_score(y_test, y_pred, average='binary')
    }

# Cross-validation scores
rf_cv_scores = cross_val_score(rf_pipeline, X, y, cv=5, scoring='f1')
xgb_cv_scores = cross_val_score(best_model, X, y, cv=5, scoring='f1')
print("Random Forest CV F1 (mean ± std):", rf_cv_scores.mean(), "±", rf_cv_scores.std())
print("XGBoost CV F1 (mean ± std):", xgb_cv_scores.mean(), "±", xgb_cv_scores.std())

# Train and evaluate
rf_pipeline.fit(X_train, y_train)
rf_results = evaluate_model(rf_pipeline, X_test, y_test)
xgb_results = evaluate_model(best_model, X_test, y_test)

print("Random Forest Test Results:", rf_results)
print("XGBoost Test Results:", xgb_results)
print("Best XGBoost Parameters:", grid_search.best_params_)

# Feature importance
feature_names = (numerical_cols.tolist() + 
                 best_model.named_steps['preprocessor'].transformers_[1][1].get_feature_names_out(categorical_cols).tolist())
selected_mask = best_model.named_steps['selector'].get_support()
selected_features = [feature_names[i] for i in range(len(feature_names)) if selected_mask[i]]
importances = best_model.named_steps['classifier'].feature_importances_
feature_imp = pd.DataFrame({'Feature': selected_features, 'Importance': importances}).sort_values(by='Importance', ascending=False)
print("\nTop 5 Feature Importances (XGBoost):\n", feature_imp.head())

# Save models
import joblib
joblib.dump(best_model, '/content/churn_model.pkl')
joblib.dump(label_encoder, '/content/label_encoder.pkl')
print("Model and label encoder saved to '/content/' directory.")