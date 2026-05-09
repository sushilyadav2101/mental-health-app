import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load dataset
df = pd.read_csv("student_mental_health.csv")

# Features
features = [
    "Sleep_Hours",
    "GPA",
    "Stress_Level",
    "Anxiety_Score",
    "Steps_Per_Day"
]

X = df[features]
y = df["Mental_Health_Status"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = RandomForestClassifier(n_estimators=150, random_state=42)
model.fit(X_train, y_train)

# Save model
pickle.dump(model, open("model.pkl", "wb"))

print("Model trained successfully!")