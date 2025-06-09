
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Load crash dataset
data = pd.read_csv("crash_data.csv")

# Features and target
features = ['speed', 'vehicle_weight', 'impact_force', 'brake_response_time']
X = data[features]
y = data['severity']  # Assume severity is labeled as 0 = Low, 1 = Medium, 2 = High

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Model training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions and evaluation
y_pred = model.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Save model predictions
pred_df = X_test.copy()
pred_df["actual_severity"] = y_test
pred_df["predicted_severity"] = y_pred
pred_df.to_csv("crash_severity_predictions.csv", index=False)
