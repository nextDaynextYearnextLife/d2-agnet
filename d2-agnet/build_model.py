import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
import numpy as np

# Load the processed data
df = pd.read_csv('processed_matches.csv')

print(f"Dataset shape: {df.shape}")
print(f"Columns: {df.columns[:10].tolist()} ... ({len(df.columns)} total)")

# Features and target variable
X = df.drop(columns=['radiant_win'])
y = df['radiant_win']

print(f"\nFeature count: {X.shape[1]}")
print(f"Samples: {X.shape[0]}")
print(f"Radiant win rate: {y.mean():.2%}")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Random Forest Classifier
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=3,
    min_samples_leaf=1,
    class_weight='balanced',
    random_state=42
)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Evaluate the model
print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.2%}")
print(f"\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print(f"\nClassification Report:\n{classification_report(y_test, y_pred, target_names=['Dire Win', 'Radiant Win'])}")

# Feature importance analysis
importances = model.feature_importances_
feature_names = X.columns
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': importances
}).sort_values('importance', ascending=False)

print("\n=== TOP 20 FEATURE IMPORTANCE ===")
print(importance_df.head(20).to_string(index=False))

# Hero-specific analysis
hero_cols = [c for c in feature_names if c.startswith('hero_')]
if hero_cols:
    hero_importance = importance_df[importance_df['feature'].isin(hero_cols)]
    print("\n=== TOP 10 MOST INFLUENTIAL HEROES ===")
    print(hero_importance.head(10).to_string(index=False))

# Save the model
joblib.dump(model, 'random_forest_model.pkl')

print("\nModel built and saved to random_forest_model.pkl")
