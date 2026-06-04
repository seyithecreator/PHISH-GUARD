import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from features import extract_features

# 1. Load your balanced dataset
df = pd.read_csv("dataset.csv")

print("🔄 Extracting 30 features from your URLs... please wait.")
X = [extract_features(url) for url in df['url']]
y = df['label']

# 2. Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Train high-performance Random Forest
print("🧠 Training the Smart Brain...")
rf = RandomForestClassifier(n_estimators=500, criterion='entropy', n_jobs=-1, random_state=42)
rf.fit(X_train, y_train)

# 4. Save
joblib.dump(rf, 'phish_model.pkl')
acc = rf.score(X_test, y_test) * 100
print(f"✅ DONE! New Accuracy: {acc:.2f}%")