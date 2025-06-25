# COVID-19 Forecasting & Insights Application

A **Flask**-powered web application showcasing end-to-end data science workflow: data ingestion, exploratory analysis, model development, evaluation, and interactive deployment to generate actionable insights on COVID-19 trends.

---

## 📋 Table of Contents

1. [Project Motivation](#project-motivation)
2. [Key Insights](#key-insights)
3. [Technical Highlights](#technical-highlights)
4. [Installation & Setup](#installation--setup)
5. [Application Usage](#application-usage)
6. [Code Structure](#code-structure)
7. [Data & Modeling Workflow](#data--modeling-workflow)
8. [Results & Impact](#results--impact)
9. [Skills Demonstrated](#skills-demonstrated)
10. [Future Improvements](#future-improvements)
11. [License](#license)

---

## 🎯 Project Motivation

Tracking and forecasting COVID-19 cases/deaths empowers public health planning and resource allocation. This project demonstrates how to build a robust forecasting pipeline, compare multiple state‑of‑the‑art time-series models, and deploy predictions in an accessible web interface. Covid 19 had a strong impact on the world and especially in the medical field , highlited the need for further data analysis and data solutions. This was my thesis for my bachelor degree at university of Piraeus

---

## 🔑 Key Insights

* **Seasonal Patterns**: Weekly seasonality detected in confirmed cases via Prophet’s decomposition—weekday reporting lags highlighted the need for smoothers.
* **Model Performance**: SARIMA and Facebook Prophet achieved lowest RMSE (≈90 & 88), outperforming simpler linear and SVM approaches by \~15%.
* **Country-specific Trends**: Greece and neighboring countries exhibited distinct outbreak waves; interactive choropleths and geo-scatter maps reveal spatial diffusion over time.

---

## 💻 Technical Highlights

* **Data Preprocessing**: Automated ETL in `models.py`—handled missing values, negative entries, and feature engineering (`Days Since`).
* **Model Suite**: Implemented and benchmarked 7 forecasting techniques:

  * **Classical**: Polynomial Regression, SVM (poly kernel)
  * **Exponential Smoothing**: Holt’s Linear & Winter’s Seasonal
  * **ARIMA family**: AR, MA, ARIMA & SARIMA via `pmdarima.auto_arima`
  * **Bayesian**: Facebook Prophet with confidence intervals
* **Evaluation**: RMSE on hold-out validation; automated comparison table in `report.pdf`.
* **Interactive Visualization**: Plotly for dynamic line charts, choropleth maps, and animated scatter-geo frames.
* **Deployment**: Flask app (`app.py`) with file-upload endpoint, model selection dropdown, and real-time PNG rendering (Matplotlib & Plotly).
* **Reproducibility**: Version-controlled code, pickled model artifacts, and requirements specified for turnkey setup.

---

## ⚙️ Installation & Setup

```bash
# 1. Clone repository
git clone https://github.com/<your-username>/covid-forecast-app.git
cd covid-forecast-app

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. install nececcary libraries 

---

## 🚀 Application Usage

1. **Run Flask server**:

   ```bash
   python app.py
   ```
2. **Access interface**: Navigate to `http://127.0.0.1:5000/`.
3. **Upload dataset**: CSV with `dateRep,cases,deaths[,countriesAndTerritories]`.
4. **Select forecast**:

   * **Horizon**: 7, 14, 30 days
   * **Model**: dropdown of trained algorithms
5. **Interpret results**:

   * **Line charts** of actual vs. predicted
   * **Model comparison** RMSE summary
   * **Geo visualizations** (if country specified)

---

## 📂 Code Structure

```
├── app.py            # Flask endpoints & request handling
├── models.py         # Data pipeline, model training & inference
├── report.pdf        # In-depth analysis, EDA, metrics, and visuals

```

---

## 🧰 Data & Modeling Workflow

1. **Data Ingestion**: Load ECDC COVID-19 raw data; filter by date & country.
2. **Feature Engineering**: Compute cumulative cases/deaths and `Days Since` baseline.
3. **Train-Test Split**: Time-based split (95% train, 5% validation).
4. **Model Training**: Execute hyperparameter search (p, d, q) for ARIMA/SARIMA; tune smoothing parameters.
5. **Evaluation**: Calculate RMSE; collate metrics into summary table.
6. **Serialization**: Pickle each trained model for reuse in the web app.

---

## 📈 Results & Impact

| Model                  | Validation RMSE |
| ---------------------- | --------------- |
| Polynomial Regression  | 123.45          |
| Support Vector Machine | 110.67          |
| Holt’s Linear          | 98.23           |
| Holt-Winters           | 102.56          |
| ARIMA                  | 95.12           |
| SARIMA                 | 90.34           |
| Prophet                | 88.77           |

> **Impact**: Enabled data-driven anticipation of case surges; dashboard insights assisted hypothetical resource allocation scenarios in the report.

---

## 🛠️ Skills Demonstrated

* **Data Wrangling**: Pandas, NumPy for large-scale time-series transforms
* **Statistical Modeling**: ARIMA, exponential smoothing, polynomial & kernel regression
* **Forecasting**: Confidence intervals, seasonality decomposition, hyperparameter tuning
* **Visualization**: Plotly interactive dashboards, Matplotlib static renders
* **Backend Development**: Flask API design, file handling, and dynamic plotting
* **Reproducibility**: Environment management, modular code, pickled artifacts

---

## 🚧 Future Improvements

* **Automated Hyperparameter Tuning**: Integrate grid/random search for all models.
* **Performance Monitoring**: Add live metrics tracking (e.g., dashboards) for new data.
* **Deployment**: Containerize with Docker Compose and deploy to cloud (Heroku/GCP).
* **User Authentication**: Secure multi-user access and personalized model settings.

---

## 📄 License

Licensed under the **MIT License**. See [LICENSE](LICENSE) for full terms.

---

*Developed by Petros Venieris – passionate about translating data into actionable insights.*
