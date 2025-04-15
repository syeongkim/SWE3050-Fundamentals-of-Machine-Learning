from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score
import pandas as pd
import numpy as np

# Load & preprocess
df = pd.read_csv("/train_data.csv")
df.drop(columns=["sku"], inplace=True)

binary_cols = ['potential_issue', 'deck_risk', 'oe_constraint', 'ppap_risk', 'stop_auto_buy', 'rev_stop']
df[binary_cols] = df[binary_cols].astype(int)

# Feature engineering
df["inv_to_sales_1m"] = df["national_inv"] / (df["sales_1_month"] + 1)
df["inv_to_forecast_3m"] = df["national_inv"] / (df["forecast_3_month"] + 1)
df["sales_ratio_1_3"] = df["sales_1_month"] / (df["sales_3_month"] + 1)
df["sales_ratio_3_6"] = df["sales_3_month"] / (df["sales_6_month"] + 1)
df["risk_score"] = df[binary_cols].sum(axis=1)
df["adjusted_lead_time"] = df["lead_time"] / (df["perf_6_month_avg"] + 0.01)

X = df.drop(columns=["went_on_backorder"])
y = df["went_on_backorder"]

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# K-Fold CV
thresholds = [0.4, 0.5, 0.6]
kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
results = []

for threshold in thresholds:
    f1s, precisions, recalls, accs, aucs = [], [], [], [], []

    for train_idx, val_idx in kfold.split(X_scaled, y):
        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = LogisticRegression(class_weight='balanced', solver='saga', max_iter=500)
        model.fit(X_train, y_train)

        probs = model.predict_proba(X_val)[:, 1]
        preds = (probs >= threshold).astype(int)

        f1s.append(f1_score(y_val, preds))
        precisions.append(precision_score(y_val, preds))
        recalls.append(recall_score(y_val, preds))
        accs.append(accuracy_score(y_val, preds))
        aucs.append(roc_auc_score(y_val, probs))

    results.append({
        "threshold": threshold,
        "f1": np.mean(f1s),
        "precision": np.mean(precisions),
        "recall": np.mean(recalls),
        "accuracy": np.mean(accs),
        "roc_auc": np.mean(aucs)
    })

results_df = pd.DataFrame(results)
print(results_df)