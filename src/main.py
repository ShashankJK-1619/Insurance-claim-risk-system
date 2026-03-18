from data_preprocessing import load_and_preprocess_data
from train_model import train_model
from evaluate_model import evaluate_model

def main():
    filepath = "data/insurance_claims.csv"

    X_train, X_test, y_train, y_test, feature_names = load_and_preprocess_data(filepath)
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test, feature_names)

if __name__ == "__main__":
    main()