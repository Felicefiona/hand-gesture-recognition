import csv
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

labels, features = [], []
with open("gesture_data.csv") as f:
    for row in csv.reader(f):
        labels.append(row[0])
        features.append([float(v) for v in row[1:]])

X = np.array(features)
y = np.array(labels)

print("Total samples:", len(y))
for name in sorted(set(y)):
    print(name, (y == name).sum())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, pred))
print(classification_report(y_test, pred))

joblib.dump(model, "gesture_model.pkl")
print("Model saved as gesture_model.pkl")