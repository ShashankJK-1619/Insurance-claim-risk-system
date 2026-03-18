import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath)

    df = df.dropna()

    X = df.drop("fraud_reported", axis=1)
    y = df["fraud_reported"].map({"Y": 1, "N": 0})

    X = pd.get_dummies(X)

    feature_names = X.columns

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test, feature_names