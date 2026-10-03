# 🚗 UK Used Car Price Predictor

A machine-learning web app that estimates the price of a used car in the UK based on its specifications. Built as a portfolio project to demonstrate end-to-end ML skills: data cleaning, exploratory analysis, model selection, and deployment.

<!-- TODO: replace with your actual Streamlit Cloud URL -->
**[▶ Try the live demo](https://YOUR-APP-NAME.streamlit.app)**

![Streamlit](https://img.shields.io/badge/Streamlit-1.45-FF4B4B?logo=streamlit)
![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python)
![XGBoost](https://img.shields.io/badge/XGBoost-3.4-189FDD)

---

## Dataset

**[100,000 UK Used Car Data set](https://www.kaggle.com/datasets/adityadesai13/used-car-dataset-ford-and-mercedes)** from Kaggle — one CSV per brand (Audi, BMW, Ford, Hyundai, Mercedes, Škoda, Toyota, Vauxhall, VW).

> The raw data is **not** committed to this repo (see `data/` in `.gitignore`).
> Download the dataset from Kaggle and place the CSVs in `data/` to reproduce the notebook.

---

## Data Cleaning

| Step | Detail | Rows |
|------|--------|------|
| Raw data loaded | Concatenated 9 brand CSVs | ~100,000 |
| Removed duplicates | Exact-match duplicates dropped **before** train/test split to avoid data leakage | — |
| Merged tax columns | Some CSVs had `tax` and others `tax(£)`; unified into a single `tax` column | — |
| Removed impossible values | Dropped rows with engine size = 0, year < 1990, mileage = 0 on old cars, etc. | — |
| Final clean dataset | Ready for modelling | **97,441** |

Full details are in [`notebooks/01_cleaning_eda.ipynb`](notebooks/01_cleaning_eda.ipynb).

---

## Key Findings from EDA

<!-- TODO: add or refine these bullet points after reviewing your notebook plots -->
- Price distributions are right-skewed; most cars sell for under £25,000.
- Mileage and year are the strongest individual predictors of price.
- Diesel and automatic cars tend to command higher prices.
- Brand has a significant effect — Mercedes, BMW and Audi sit above the dataset average.

---

## Model Comparison

All models use a **scikit-learn Pipeline** with `OneHotEncoder` for categorical features and the regressor. Evaluated on a **20 % held-out test set** (random split after duplicate removal).

| Model | Test R² | Test MAE (£) |
|-------|---------|-------------|
| Linear Regression (baseline) | 0.785 | £2,860 |
| Random Forest | 0.964 | £1,128 |
| **XGBoost** ✅ | **0.963** | **£1,201** |

The **Random Forest** achieved slightly better MAE, but XGBoost was chosen for deployment because it produces a smaller, faster model with comparable performance.

### Feature Importance

<!-- TODO: paste or link a feature importance plot here -->
<!-- e.g. ![Feature importance](assets/feature_importance.png) -->

*Placeholder — add a bar chart of XGBoost feature importances from the notebook.*

---

## Repository Structure

```
.
├── app.py                          # Streamlit web app
├── requirements.txt                # Pinned Python dependencies
├── models/
│   ├── price_model.joblib          # Trained XGBoost pipeline (~584 KB)
│   └── options.json                # UI options & metadata exported by notebook
├── notebooks/
│   └── 01_cleaning_eda.ipynb       # Data cleaning, EDA & model training
├── data/                           # Raw CSVs (git-ignored)
├── .gitignore
└── README.md
```

---

## Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/uk-used-car-price-predictor.git
cd uk-used-car-price-predictor

# 2. Create a virtual environment (optional but recommended)
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the app
streamlit run app.py
```

The app will open at `http://localhost:8501`.

> **Note:** You do **not** need the raw data to run the app — the trained model is included in `models/`.

---

## Limitations

- **UK prices only** — the model was trained on UK listings priced in £. It will not give meaningful estimates for other markets.
- **Data period: 1996–2020** — cars newer than 2020 or older than 1996 are outside the training range.
- **Rare combinations are less reliable** — predictions for unusual brand/model/fuel combinations (e.g. a diesel Mustang) will be less accurate because the model saw few or no similar examples.
- **No inflation adjustment** — prices reflect the market at the time the data was scraped, not current values.

---

## What I Would Improve

- Collect more recent data (2021+) to keep the model current.
- Add hyperparameter tuning (e.g. Optuna / RandomizedSearchCV).
- Experiment with target-encoding instead of one-hot to reduce dimensionality.
- Add SHAP-based explanations to the Streamlit app so users can see *why* a price was estimated.
- Track model performance over time with experiment logging (MLflow / Weights & Biases).

---

## License

This project is for educational / portfolio purposes. The dataset is provided by [Aditya Desai on Kaggle](https://www.kaggle.com/datasets/adityadesai13/used-car-dataset-ford-and-mercedes) under its own licence.
