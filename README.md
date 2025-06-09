Customer Churn Prediction
Overview
This project predicts whether telecom customers will churn (leave the service) using machine learning. It uses a dataset (customer_churn_data.csv) with customer details like tenure, services, and charges. Two models are implemented: Random Forest and XGBoost, with hyperparameter tuning and feature selection to improve performance.
Dataset

File: customer_churn_data.csv (not included in the repository due to size; users must provide it)
Columns: Includes gender, tenure, MonthlyCharges, TotalCharges, Churn, and service-related features.
Target: Churn (Yes/No)

Project Structure
customer-churn-prediction/
├── churn_prediction.py        # Main script for data processing, model training, and evaluation
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
├── .gitignore                 # Files to ignore in Git

Note: customer_churn_data.csv, churn_model.pkl, and label_encoder.pkl are not included in the repository. Place customer_churn_data.csv in the project folder before running the script.
Installation

Clone the repository:git clone https://github.com/KindaPINK-prAF/customer-churn-prediction.git
cd customer-churn-prediction


Create and activate a virtual environment:python -m venv venv
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows


Install dependencies:pip install -r requirements.txt


Place customer_churn_data.csv in the project folder.

Usage

Run the script:python churn_prediction.py


The script will:
Load and preprocess the data
Train Random Forest and XGBoost models
Tune XGBoost hyperparameters
Evaluate models using accuracy, precision, recall, and F1 score
Save the trained model (churn_model.pkl) and label encoder (label_encoder.pkl)



Results

Random Forest: Uses balanced class weights and a max depth of 5.
XGBoost: Optimized with GridSearchCV for best parameters.
Metrics: Reports accuracy, precision, recall, and F1 score.
Feature Importance: Shows top features affecting churn.

Dependencies
Listed in requirements.txt. Key libraries:

pandas
numpy
scikit-learn
xgboost
joblib

Future Improvements

Add plots for feature importance and model performance
Try other models like Neural Networks or LightGBM
Build a web app to deploy the model using Flask

Author

[Priyanshu Sarraf]
Email: [psharraf777@gmail.com]
GitHub: [KindaPINK-prAF]

License
This project is licensed under the MIT License.
