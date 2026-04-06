# Agro-Insight — AI-Powered Crop Recommendation System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-Backend-000000?style=for-the-badge&logo=flask&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Bootstrap](https://img.shields.io/badge/Bootstrap-Frontend-7952B3?style=for-the-badge&logo=bootstrap&logoColor=white)
![Accuracy](https://img.shields.io/badge/Model%20Accuracy-99.3%25-brightgreen?style=for-the-badge)

*Empowering farmers with data-driven decisions — one crop at a time.*

</div>

---

## Overview

**Agro-Insight** is a full-stack machine learning web application built to assist farmers and agricultural professionals in choosing the most suitable crop for their land. By analyzing **7 key soil and atmospheric parameters**, the system intelligently recommends the best crop out of **22 possible options** — helping maximize yield and profitability.

This project showcases a complete end-to-end ML pipeline: from raw data exploration and model training to real-time predictions via a clean, interactive Flask web interface.

---

## Key Features

| Feature | Description |
|---|---|
| **Smart Recommendations** | Random Forest Classifier with **99.3% test accuracy** across 22 crop types |
| **7 Environmental Inputs** | Soil NPK, Temperature, Humidity, pH, and Rainfall |
| **Interactive Web UI** | Modern, responsive interface built with HTML, Bootstrap & JavaScript |
| **Real-Time Predictions** | Flask backend serves instant crop predictions on input submission |
| **Documented Research** | Full model comparison documented in a Jupyter Notebook |

---

## Input Parameters

The model uses the following environmental features to make its prediction:

```
N  → Nitrogen content in soil
P  → Phosphorus content in soil
K  → Potassium content in soil
T  → Temperature (°C)
H  → Humidity (%)
pH → Soil pH level
R  → Rainfall (mm)
```

---

## Tech Stack

```
├── Backend       → Python, Flask
├── ML & Data     → Scikit-learn, Pandas, NumPy
├── Frontend      → HTML5, CSS3, Bootstrap, JavaScript
└── Exploration   → Jupyter Notebook
```

---

## Model Performance

Ten different classification models were benchmarked during development. The **Random Forest Classifier** was selected for its superior performance:

| Model | Accuracy |
|---|---|
| Random Forest Classifier | **99.3%** |
| Other models tested | < 99.3% |

The full comparison and evaluation process is documented in:
> `Agroinsight.ipynb`

---

## Getting Started

Follow the steps below to set up and run the project locally.

### Prerequisites

- Python 3.7+
- [Anaconda](https://www.anaconda.com/download) or Miniconda (recommended)

---

### Step 1 — Create & Activate Conda Environment

```bash
conda create --name cropprediction python=3.9 -y
conda activate cropprediction
```

### Step 2 — Install Dependencies

```bash
pip install Flask scikit-learn pandas numpy
```

### Step 3 — Navigate to Project Directory

```bash
cd "your/project/folder/path"
```

### Step 4 — Train the Model

```bash
python train_model.py
```

> On success, you will see: `Model and scaler have been saved successfully...`
>
> This generates two files: `model.pkl` and `minmaxscaler.pkl`

### Step 5 — Launch the Flask App

```bash
python appnew.py
```

### Step 6 — Open in Browser

```
http://127.0.0.1:5000
```

---

## Project Structure

```
Agro-Insight/
│
├── models/                       # Saved model files (model.pkl, scaler.pkl)
├── src/                          # Source code (Flask app, training scripts)
├── templates/                    # HTML templates for the web UI
│
├── Agroinsight.ipynb             # Jupyter Notebook (EDA + model comparison)
├── Crop_recommendation.csv       # Dataset used for training
├── LICENSE                       # Project license
├── .gitignore
└── README.md
```

---

## Supported Crops

The system can recommend from **22 different crops**, covering a wide variety of agricultural produce suited to different soil and climate conditions.

---

## Contributions

 **Ayush Sharma** — Lead, model training and full-stack deployment  
 **Anubhab Halder** — Backend development and pipeline integration  
 **Yashodhara Singh** — Testing, debugging and maintainence

