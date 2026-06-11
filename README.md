# 🏠 House Price Predictor

A machine learning web app that predicts house sale prices using Ridge Regression, trained on the Kaggle House Prices dataset.

**[🚀 Live Demo →](your-streamlit-link-here)**

---

## 📊 Model Performance
| Metric | Score |
|--------|-------|
| R² Score | ~0.87 |
| Cross-val R² (5-fold) | ~0.85 |
| Algorithm | Ridge Regression (α=10) |
| Training samples | 1,456 |
| Features used | 20 |

---

## 🛠️ Tech Stack
`Python` `scikit-learn` `Pandas` `NumPy` `Streamlit` `Matplotlib` `Seaborn` `Joblib`

---

## 🔍 Project Highlights
- **EDA** — distribution analysis, correlation heatmap, outlier detection
- **Feature Engineering** — created 7 new features (TotalSF, TotalBath, HouseAge, etc.)
- **Model Comparison** — Linear Regression vs Ridge; Ridge selected for lower variance
- **Deployed App** — interactive Streamlit UI with live predictions and feature importance chart

---

## 📁 Project Structure
```
house-price-predictor/
├── house_price_day1.py     # EDA + training pipeline
├── app.py                  # Streamlit web app
├── model.pkl               # Trained Ridge model
├── scaler.pkl              # StandardScaler
├── features.pkl            # Feature list
├── requirements.txt
└── README.md
```

---

## ▶️ Run Locally
```bash
git clone https://github.com/yourusername/house-price-predictor
cd house-price-predictor
pip install -r requirements.txt

# 1. Download train.csv from Kaggle and place it here
# 2. Train the model
python house_price_day1.py

# 3. Launch the app
streamlit run app.py
```

---

## 📈 Key Findings
- **Overall Quality** and **Total Square Footage** are the strongest price predictors
- Log-transforming SalePrice improved model fit significantly
- Ridge regularization reduced overfitting vs plain Linear Regression

---

*Built as part of a hands-on ML learning journey covering statistics, feature engineering, and regression algorithms.*
