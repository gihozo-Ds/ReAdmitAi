# 🏥 ReAdmitAi: Patient Readmission Predictor

## Overview

ReAdmitAi is a machine learning project that predicts whether a patient will be readmitted to the hospital within 30 days, using the [Diabetes 130-US hospitals dataset](https://archive.ics.uci.edu/ml/datasets/Diabetes+130-US+hospitals+for+years+1999-2008#).  
It includes a Streamlit web app for easy predictions and Jupyter notebooks for data analysis and model building.

---

## Features

- Data cleaning and preprocessing
- Exploratory Data Analysis (EDA)
- Logistic Regression model for readmission prediction
- Feature importance visualization
- Streamlit web app for user-friendly predictions

---

## Project Structure

```
ReAdmitAi/
│
├── data/
│   └── diabetic_data.csv
├── notebooks/
│   └── Untitled-1.ipynb
├── dashboard/
│   └── app.py
├── requirements.txt
└── README.md
```

---

## Setup & Installation

1. **Clone the repository:**
   ```sh
   git clone <your-repo-url>
   cd ReAdmitAi
   ```

2. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

3. **Download the dataset:**
   - Place `diabetic_data.csv` in the `data/` folder.

4. **Run the Streamlit app:**
   ```sh
   cd dashboard
   streamlit run app.py
   ```

---

## Usage

- Use the Streamlit app to input patient data and get a readmission prediction.
- Explore the Jupyter notebook in `notebooks/` for data analysis and model details.

---

## Requirements

- Python 3.8+
- See `requirements.txt` for all dependencies.

---

## Model Training

- Data cleaning, EDA, and model training are in `notebooks/Untitled-1.ipynb`.
- The trained model is saved as `readmission_model.pkl` and used by the Streamlit app.

---

## Acknowledgements

- [UCI Machine Learning Repository: Diabetes 130-US hospitals dataset](https://archive.ics.uci.edu/ml/datasets/Diabetes+130-US+hospitals+for+years+1999-2008#)

---

## License

MIT License
