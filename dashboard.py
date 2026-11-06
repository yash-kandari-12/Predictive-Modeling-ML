import pandas as pd
import joblib
import plotly.express as px
import webbrowser, os

info = joblib.load("model_information.pkl")
eval_df = pd.read_csv("improved_model_evaluation.csv")
pred_df = pd.read_csv("demand_predictions.csv")

models = ["Linear_Regression", "Decision_Tree", "Random_Forest"]
figs = []

# 1. Model comparison
name_col = eval_df.columns[0]
long = eval_df.melt(id_vars=name_col, var_name="Metric", value_name="Value")
figs.append(px.bar(long, x=name_col, y="Value", color="Metric",
                   barmode="group", title="Model Comparison"))

# 2. Actual vs predicted (first 200 rows)
sample = pred_df.head(200)
figs.append(px.line(sample, y=["Actual_Demand"] + models,
                    title="Actual vs Predicted Demand (first 200 rows)"))

# 3. Scatter for each model
for m in models:
    figs.append(px.scatter(pred_df, x="Actual_Demand", y=m, opacity=0.5,
                           title=f"Actual vs Predicted: {m}"))

# 4. Residuals for each model
for m in models:
    resid = pred_df["Actual_Demand"] - pred_df[m]
    figs.append(px.histogram(resid, nbins=50, title=f"Residuals: {m}"))

# Build one HTML page
html = f"""<html><head><meta charset="utf-8"><title>Demand Prediction Dashboard</title></head>
<body style="font-family:Arial;margin:30px">
<h1>📈 Demand Prediction Dashboard</h1>
<h3>Best model: {info['model_name']} &nbsp;|&nbsp; Features used: {len(info['features'])}</h3>
<h3>Evaluation table</h3>{eval_df.to_html(index=False)}
"""
for i, f in enumerate(figs):
    html += f.to_html(full_html=False, include_plotlyjs=(i == 0))
html += "</body></html>"

with open("dashboard.html", "w", encoding="utf-8") as f:
    f.write(html)

webbrowser.open("file://" + os.path.abspath("dashboard.html"))
print("Dashboard created: dashboard.html")
