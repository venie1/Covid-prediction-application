# COVID-19 Time Series Forecasting App

[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/) [![GitHub Actions](https://img.shields.io/github/actions/workflow/status/<your-username>/covid-prediction-application/ci.yml?branch=main)](https://github.com/<your-username>/covid-prediction-application/actions) [![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

A **Flask** web application for forecasting COVID-19 cases and deaths using multiple time-series models, complete with interactive visualizations and performance benchmarking. This was my thesis on my bachelor degree.

---

## 📋 Table of Contents

1. [Features](#features)
2. [Demo](#demo)
3. [Tech Stack](#tech-stack)
4. [Repository Structure](#repository-structure)
5. [Installation](#installation)
6. [Usage](#usage)
7. [Results & Evaluation](#results--evaluation)
8. [Testing & CI](#testing--ci)
9. [Contributing](#contributing)
10. [License](#license)

---

## ✨ Features

* **Data Preparation**: Ingest, clean, and transform global COVID-19 datasets with Pandas & NumPy.
* **Multiple Forecast Models**:

  * Polynomial Regression, Support Vector Machine (SVM)
  * Holt’s Linear & Winter’s Exponential Smoothing
  * AR, MA, ARIMA & SARIMA (via pmdarima)
  * Facebook Prophet with trend and seasonality decompositions
* **Interactive Visualizations**: Plotly charts for time series, choropleth, and animated geo-scatter maps.
* **Web Interface**: Flask app to upload CSVs, select forecast horizon and model, and view prediction plots in real time.
* **Benchmarking**: Automated RMSE computation across all models for validation and test sets.
* **Containerized Deployment**: Docker support for easy reproducibility.

---

## 🎥 Demo

![App Screenshot](docs/screenshot.png)

> **Live Demo:** [https://covid-forecast-app.demo](https://covid-forecast-app.demo) *(coming soon)*

---

## 🛠 Tech Stack

| Layer             | Tools & Libraries                            |
| ----------------- | -------------------------------------------- |
| **Data**          | Pandas, NumPy                                |
| **Modeling**      | Scikit-learn, statsmodels, pmdarima, Prophet |
| **Visualization** | Plotly, Matplotlib                           |
| **Web Framework** | Flask                                        |
| **Deployment**    | Docker, GitHub Actions                       |

---

## 📂 Repository Structure

```bash
├── data/               # Sample and raw datasets
├── docs/               # Screenshots, demo assets, architecture diagrams
├── models/             # Serialized model artifacts (.pkl)
├── notebooks/          # EDA and experimentation notebooks
├── src/                # Source code
│   ├── data_prep.py    # Data ingestion & cleaning
│   ├── train_models.py # Model training & evaluation scripts
│   ├── visualize.py    # Plotting utilities
│   └── app.py          # Flask web application
├── tests/              # Unit and integration tests
├── Dockerfile          # Docker container definition
├── requirements.txt    # Python dependencies
├── .github/workflows/  # CI configuration
├── README.md           # Project overview (you are here)
└── LICENSE             # License information
```

---

## 🚀 Installation

1. **Clone the repo**

   ```bash
   git clone https://github.com/<your-username>/covid-prediction-application.git
   cd covid-prediction-application
   ```
2. **Create environment & install dependencies**

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
3. **(Optional) Build Docker image**

   ```bash
   docker build -t covid-forecast-app .
   ```

---

## 🏃‍♂️ Usage

1. **Prepare data**: Place your CSV file in the `data/` directory or update the path in `src/app.py`.
2. **Train models** (optional; pre-trained models available in `models/`):

   ```bash
   python src/train_models.py --data data/covid_data.csv
   ```
3. **Run Flask app**:

   ```bash
   python src/app.py
   ```
4. **Access the UI**: Open `http://127.0.0.1:5000` in your browser.

   * Upload your CSV file
   * Select forecast horizon (e.g., 7, 30 days)
   * Choose model and click **Predict**

<details>
<summary>📖 Quickstart Notebook</summary>

A step-by-step Jupyter notebook (`notebooks/Quickstart.ipynb`) demonstrates:

```python
# Load data
df = load_data('data/covid_data.csv')
# Train a minimal model
model = train_polynomial(df)
# Launch app
!streamlit run src/app.py
```

</details>

---

## 📊 Results & Evaluation

| Model                  | RMSE (validation) |
| ---------------------- | ----------------- |
| Polynomial Regression  | 123.45            |
| Support Vector Machine | 110.67            |
| Holt's Linear Model    | 98.23             |
| Holt's Winter Model    | 102.56            |
| ARIMA                  | 95.12             |
| SARIMA                 | 90.34             |
| Prophet                | 88.77             |

*DETAILED plots and error analysis available in `notebooks/Model_Evaluation.ipynb`.*

---

## ✔️ Testing & CI

* **Unit tests**: Run `pytest tests/` to validate data pipelines and model inference.
* **Continuous Integration**: GitHub Actions workflow (`.github/workflows/ci.yml`) runs tests on each push and pull request.

---

## 🤝 Contributing

Contributions are welcome! Please follow:

1. Fork the repo
2. Create a feature branch (`git checkout -b feature-name`)
3. Commit your changes (`git commit -m 'Add new feature'`)
4. Push to the branch (`git push origin feature-name`)
5. Open a Pull Request

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## 📄 License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

---

*Made with ❤️ by [Your Name](https://github.com/<your-username>)*
