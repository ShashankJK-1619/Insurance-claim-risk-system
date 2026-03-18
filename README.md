# 🛡️ Insurance Claim Risk Prediction System

## 📌 Overview

This project is an end-to-end **Machine Learning system** that predicts whether an insurance claim is **high risk (potential fraud)** or **low risk** based on customer and claim details.

It includes:

* Data preprocessing
* Model training and evaluation
* A deployed **Streamlit web app**
* Probability-based fraud risk prediction

---

## 🚀 Features

* Predicts fraud risk using a trained ML model
* Displays **fraud probability (%)**
* Uses **custom threshold tuning** for better fraud detection
* Interactive web UI built with Streamlit
* Modular ML pipeline (preprocessing → training → evaluation → app)

---

## 🧠 Machine Learning Approach

### Model Used

* Random Forest Classifier

### Key Improvements

* Handled class imbalance using:

  * `class_weight` tuning
* Improved fraud detection using:

  * **Probability thresholding (0.30 instead of default 0.50)**

---

## 📊 Model Performance

```
Accuracy: ~70%

Class 0 (Low Risk):
Precision: 0.71
Recall: 0.98

Class 1 (High Risk / Fraud):
Precision: 0.50
Recall: 0.06
```

### ⚠️ Important Insight

Fraud detection is a **class imbalance problem**.
Instead of maximizing accuracy, this project focuses on:

> Increasing fraud sensitivity using probability thresholds.

---

## 🧪 How It Works

1. Raw data is preprocessed and encoded
2. Model is trained using processed features
3. Streamlit app:

   * Collects user inputs
   * Matches model feature structure
   * Predicts fraud probability
4. Decision logic:

```python
if probability > 0.30:
    High Risk
else:
    Low Risk
```

---

## 💻 Run Locally

### 1. Clone the repo

```bash
git clone https://github.com/ShashankJK-1619/insurance-claim-risk-system.git
cd insurance-claim-risk-system
```

### 2. Create virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Train model

```bash
python src/main.py
```

### 5. Run app

```bash
streamlit run app/streamlit_app.py
```

---

## 📁 Project Structure

```
insurance-claim-risk-system/
│
├── app/
│   └── streamlit_app.py
│
├── data/
│   └── insurance_claims.csv
│
├── models/
│   └── claim_model.pkl
│
├── src/
│   ├── data_preprocessing.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   └── main.py
│
├── requirements.txt
└── README.md
```

---

## 🔍 Key Learnings

* Handling **class imbalance** is critical in fraud detection
* Accuracy is not enough → **recall and thresholds matter**
* Real-world ML apps require:

  * Feature consistency between training and inference
  * Proper preprocessing pipelines
* Deploying models requires careful input alignment

---

## 🌐 Future Improvements

* Use SMOTE or advanced resampling techniques
* Save preprocessing pipeline with model (Pipeline)
* Add feature importance visualization
* Deploy on Streamlit Cloud / AWS
* Improve UI with dashboards and charts

---

## Live Demo
👉 https://insurance-claim-risk-system-4lefty4a2ua62tkjsjcpym.streamlit.app

## Author
Shashank Jayakumar

## Project Links
- GitHub Repo: https://github.com/ShashankJK-1619/Insurance-claim-risk-system
- Live App: https://insurance-claim-risk-system-4lefty4a2ua62tkjsjcpym.streamlit.app
