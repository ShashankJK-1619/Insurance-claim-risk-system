from sklearn.ensemble import RandomForestClassifier
import joblib
import os

def train_model(X_train, y_train):
    os.makedirs("models", exist_ok=True)

    model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight={0:1, 1:4}  
     )

    model.fit(X_train, y_train)
    joblib.dump(model, "models/claim_model.pkl")

    print("Model trained and saved.")
    return model