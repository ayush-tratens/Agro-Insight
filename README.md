Agro-Insight: AI-Powered Crop Recommendation System
Overview -->

Agro-Insight is a web-based machine learning application designed to help farmers and agricultural professionals make data-driven decisions. By analyzing key soil and atmospheric features, the system recommends the most suitable crop to cultivate from a list of 22 different options, aiming to increase yield and profitability.

This project demonstrates an end-to-end machine learning workflow, from data exploration and model training to deployment via a user-friendly web interface powered by Flask.

Key Features -->

    Intelligent Crop Recommendation: Utilizes a Random Forest Classifier to predict the optimal crop with 99.3% accuracy on the test set.

    Data-Driven Insights: Considers 7 crucial environmental factors:

        Nitrogen (N), Phosphorus (P), and Potassium (K) content in the soil.

        Temperature

        Humidity

        Soil pH

        Rainfall

    Interactive Web Interface: A clean and modern UI built with HTML and Bootstrap allows for easy input of farm data.

    Full-Stack Implementation: A complete system with a Scikit-learn model backend and a Flask server to handle real-time predictions.

TECHS USED -->

    Backend: Python, Flask

    Machine Learning: Scikit-learn, Pandas, NumPy

    Frontend: HTML, CSS, Bootstrap, JavaScript

    Data Exploration: Jupyter Notebook

HOW TO EXECUTE THE PROJECT -->

Follow these instructions to get a local copy up and running.
Prerequisites

    Python 3.7+

    Anaconda or Miniconda distribution (recommended for managing environments)

Installation & Setup

    download and install anaconda 
    open anaconda prompt -->

    conda create --name cropprediction python=3.9 -y
    conda activate cropprediction

    Install the required libraries:

    pip install Flask scikit-learn pandas numpy


	after a successful installation navigate to your project directory 
	using cd "project directory folder path "

    Train the model:
    Run the training script to generate the model.pkl and minmaxscaler.pkl files.

    python train_model.py

    You should see a success message: ✅ Model and scaler have been saved successfully...

    Run the Flask application:

    python appnew.py

    Access the application:
    Open your web browser and navigate to:
    http://127.0.0.1:5000


📈 Model Performance

During development, ten different classification models were evaluated. The Random Forest Classifier was selected for its superior performance, achieving an accuracy of 99.3% on the unseen test data. The experimentation and evaluation process is documented in the Crop Classification With Recommendation System.ipynb notebook.


## Contributors

**Ayush Sharma** — Lead, model training and full-stack deployment   
**Anubhab Halder** — Backend development and pipeline integration
