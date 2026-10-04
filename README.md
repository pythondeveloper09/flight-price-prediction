# ✈️ Flight Price Prediction

A Machine Learning based web application that predicts flight ticket prices based on different flight-related features such as airline, source, destination, travel class, departure time, arrival time, number of stops, and travel duration.

The project uses a **Decision Tree Regression model** and provides an interactive web interface using **Streamlit**.

---

## 🚀 Project Overview

Flight ticket prices depend on several factors, including:

- Airline
- Departure city
- Arrival city
- Travel class
- Departure time
- Arrival time
- Number of stops
- Travel duration

This project uses historical flight data to train a **Decision Tree Regressor** and deploys the trained model through a Streamlit application.

---

## 🎯 Problem Statement

To build a Machine Learning regression model that can predict the estimated price of a flight based on its travel and flight-related characteristics.

The trained Decision Tree model is integrated with a Streamlit web application to provide real-time flight price predictions.

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Decision Tree Regressor
- Streamlit
- Pickle

---

## 📊 Dataset

The project uses flight price datasets containing information related to Business and Economy class flights.

The dataset contains features such as:

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

Business and Economy datasets were combined during the data preparation process.

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
Decision Tree Regression
       ↓
Model Evaluation
       ↓
Model Serialization
       ↓
Streamlit Deployment
