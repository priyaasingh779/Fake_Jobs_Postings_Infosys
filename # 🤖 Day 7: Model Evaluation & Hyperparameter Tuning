# -----------------------------------
# 🤖 Day 7: Model Evaluation & Hyperparameter Tuning
# -----------------------------------

import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# 1️⃣ Load dataset
df = pd.read_csv('fake_job_postings.csv')
df = df.dropna(subset=['description'])

# 2️⃣ TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=3000)
X = vectorizer.fit_transform(df['description'])
y = df['fraudulent']

# 3️⃣ Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4️⃣ Initialize Models
log_reg = LogisticRegression(max_iter=200)
rf = RandomForestClassifier(random_state=42)

# 5️⃣ Cross-Validation (to check consistency of models)
log_cv = cross_val_score(log_reg, X_train, y_train, cv=5, scoring='accuracy')
rf_cv = cross_val_score(rf, X_train, y_train, cv=5, scoring='accuracy')
print("Logistic Regression CV Accuracy:", log_cv.mean())
print("Random Forest CV Accuracy:", rf_cv.mean())

# 6️⃣ Train both models
log_reg.fit(X_train, y_train)
rf.fit(X_train, y_train)

# 7️⃣ ROC-AUC Curve (Model Comparison)
y_prob_log = log_reg.predict_proba(X_test)[:, 1]
y_prob_rf = rf.predict_proba(X_test)[:, 1]

fpr1, tpr1, _ = roc_curve(y_test, y_prob_log)
fpr2, tpr2, _ = roc_curve(y_test, y_prob_rf)

plt.figure(figsize=(8,5))
plt.plot(fpr1, tpr1, label="Logistic Regression")
plt.plot(fpr2, tpr2, label="Random Forest")
plt.plot([0,1], [0,1], 'k--', label='Random Chance')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend()
plt.show()

print("Logistic Regression AUC:", roc_auc_score(y_test, y_prob_log))
print("Random Forest AUC:", roc_auc_score(y_test, y_prob_rf))

# 8️⃣ Hyperparameter Tuning (GridSearchCV for Random Forest)
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5],
}

grid = GridSearchCV(rf, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
grid.fit(X_train, y_train)

print("\nBest Parameters from GridSearchCV:", grid.best_params_)
print("Best Cross-Validation Accuracy:", grid.best_score_)
