# Demand Prediction using Machine Learning

A machine learning project that predicts product demand from the Online Retail dataset and shows the results in an interactive dashboard.

## Models Used
- Linear Regression
- Decision Tree
- Random Forest

Best model: <your best model name>

## Project Files
- `Machine Learning.py` - cleans the data, trains and evaluates the models, saves the best one
- `dashboard.py` - builds the interactive dashboard (`dashboard.html`)
- `dashboard.html` - final dashboard with model comparison, actual vs predicted and residual charts
- `best_demand_model.pkl`, `model_information.pkl` - saved model and its details
- `improved_model_evaluation.csv`, `demand_predictions.csv` - evaluation results and predictions

## How to Run
1. Install the libraries:
   python -m pip install pandas numpy scikit-learn joblib plotly openpyxl
2. Train the models:
   python "Machine Learning.py"
3. Create the dashboard:
   python dashboard.py
4. Open `dashboard.html` in your browser.

## Dataset
UCI Online Retail dataset.

## Tools
Python, Pandas, Scikit-learn, Plotly
