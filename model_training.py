import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os

# Load dataset
df = pd.read_csv('adult 3.csv')

# Drop missing values
df = df.dropna()

# Fix any column name mismatch
if 'educational-num' in df.columns:
    df.rename(columns={'educational-num': 'education-num'}, inplace=True)

# Features and target
X = df.drop('income', axis=1)
y = df['income']

# Encode categorical columns
encoders = {}
for col in X.columns:
    if X[col].dtype == 'object':
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        encoders[col] = le

# Encode target
y_encoder = LabelEncoder()
y = y_encoder.fit_transform(y)
encoders['income'] = y_encoder  # Save target encoder too

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"✅ Model Accuracy: {acc:.2f}")

# Save model and encoders
os.makedirs('model', exist_ok=True)
joblib.dump(model, 'model/salary_model.joblib')
joblib.dump(encoders, 'model/encoders.joblib')
