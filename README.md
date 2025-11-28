# Housing-price-prediction (/-_-\)
This project is a simple web application that predicts housing prices based on various features. It provides an easy-to-use interface for users to get an estimated price for a house with certain features.

## Problem Solved

This project can be used by real estate agents, buyers, or sellers to get a quick estimate of a property's value. It helps in making informed decisions by providing a data-driven prediction of house prices.

## How to Run the Project

1.**Clone the repository:**
    ```bash
    git clone <https://github.com/shallzy123/crispy-octo-parakeet/tree/main>
    cd <repository-directory>
    ```

2.**Create a virtual environment and install dependencies:**
    ```bash
    python -m venv venv
    venv\Scripts\activate
    pip install -r requirements.txt
    ```

3.**Run the Streamlit application:**
    ```bash
    streamlit run app.py
    ```

## How it Works

- The application is built using **Streamlit**, a Python library for creating interactive web apps.
- The machine learning model is a **Linear Regression** model trained on the `Housing.csv` dataset.
- The categorical features in the dataset are encoded using `LabelEncoder` from **scikit-learn**.
- The trained model and encoders are saved as `.pkl` files (`house_price_model.pkl` and `encoders.pkl`) using **joblib**.
- The user provides input through the Streamlit interface, which is then preprocessed and fed to the model to get the price prediction.

## Dependencies

The project's dependencies are listed in the `requirements.txt` file:

- streamlit
- pandas
- scikit-learn
- joblib
- numpy
- scipy

## Data

The data used for training the model is from the `Housing.csv` file, which contains various features of houses and their corresponding prices.
