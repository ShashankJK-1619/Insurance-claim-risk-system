from sklearn.metrics import accuracy_score, classification_report

def evaluate_model(model, X_test, y_test, feature_names):
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.2f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))