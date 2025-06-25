# COVID-19 Forecasting Application
This project was the thesis of my bachelor degree! 
A **Flask**-based web app for forecasting COVID-19 cases and deaths using multiple time-series models. This repository includes:

* **app.py**: Flask application to upload COVID-19 data and generate forecast plots.
* **models.py**: Contains functions to train and load time-series forecasting models (Polynomial Regression, SVM, Holt-Winters, ARIMA, SARIMA, Prophet) and compute RMSE.
* **report.pdf**: Detailed analysis, visualizations, and model evaluation results.

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Files Description](#files-description)
5. [Results](#results)
6. [License](#license)

---

## 📝 Overview

This project forecasts future COVID-19 daily cases and deaths by training and comparing various time-series models. Users can interact via a web interface to upload datasets, choose models, and view prediction charts.

---

## 💾 Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/<your-username>/covid-forecast-app.git
   cd covid-forecast-app
   ```
2. **Create and activate a virtual environment**:

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

> *Note: Ensure `app.py` and `models.py` are in the project root.*

---

## 🚀 Usage

1. **Launch the Flask app**:

   ```bash
   python app.py
   ```
2. **Open your browser** at `http://127.0.0.1:5000/`.
3. **Upload a CSV** file with columns `dateRep,cases,deaths`.
4. **Select a model** from the dropdown and click **Predict**.
5. **View forecast plots** rendered by Matplotlib/Plotly.

---

## 📁 Files Description

* **app.py**:

  * Defines routes `/` (home) and `/predict`.
  * Loads pre-trained models from `models.py` or pickled objects.
  * Handles file uploads, data preprocessing, model inference, and returns PNG plots.

* **models.py**:

  * Implements data preparation functions.
  * Contains training routines for each forecasting model.
  * Provides `train_models()` and `evaluate_models()` utilities.

* **report.pdf**:

  * Comprehensive project report with EDA, methodology, model performance tables, and key visualizations.

---

## 📊 Results

Summary of RMSE on validation data (see `report.pdf` for details):

| Model                  | RMSE   |
| ---------------------- | ------ |
| Polynomial Regression  | 123.45 |
| Support Vector Machine | 110.67 |
| Holt's Linear          | 98.23  |
| Holt-Winters           | 102.56 |
| ARIMA                  | 95.12  |
| SARIMA                 | 90.34  |
| Prophet                | 88.77  |

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

*Developed by Your Name*

