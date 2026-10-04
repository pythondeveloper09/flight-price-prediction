# ✈️ Flight Price Prediction

A Machine Learning based web application that predicts flight ticket prices based on flight details such as airline, source, destination, travel class, departure time, arrival time, number of stops, and travel duration.

The application is built using **Python, Scikit-learn, Random Forest Regression, and Streamlit**.

---

## 🚀 Project Overview

Flight ticket prices depend on multiple factors such as:

- Airline
- Departure and arrival locations
- Travel class
- Departure time
- Arrival time
- Number of stops
- Total travel duration

This project uses historical flight data to train a **Random Forest Regression model** and provides an interactive Streamlit application for predicting the estimated flight price.

---

## 🎯 Problem Statement

To develop a machine learning regression model that can estimate flight ticket prices based on different flight-related features.

The trained model is deployed through a simple and user-friendly **Streamlit web application**.

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Random Forest Regressor
- Streamlit
- Pickle

---

## 📊 Dataset

The project uses flight price data containing information about:

- Date
- Airline
- Flight Code
- Flight Number
- Departure Time
- Source
- Travel Duration
- Number of Stops
- Arrival Time
- Destination
- Flight Class
- Price

The Business and Economy datasets were combined during the data preparation process.

---

## 🔄 Machine Learning Workflow

```text
Raw Flight Data
       ↓
Data Cleaning
       ↓
Data Preprocessing
       ↓
Feature Engineering
       ↓
Categorical Encoding
       ↓
Train-Test Split
       ↓
Random Forest Regression
       ↓
Model Evaluation
       ↓
Model Serialization
       ↓
Streamlit Deployment
