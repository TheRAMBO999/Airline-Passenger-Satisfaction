# Airline Passenger Satisfaction System

This repository contains the implementation of an Airline Passenger Satisfaction System, which aims to analyze and predict passenger satisfaction based on various factors like flight experience, demographics, and service quality.

## Table of Contents

- [Introduction](#introduction)

- [Dataset](#dataset)
  
- [Models Used](#models-used)

- [Installation](#installation)

- [Project Workflow](#project-workflow)

- [Technologies Used](#Technologies-Used)

- [Results](#results)


## Introduction

The Airline Passenger Satisfaction System uses machine learning techniques to classify passenger satisfaction as either "Satisfied" or "Dissatisfied." By analyzing patterns in flight data, this system helps airlines improve customer experience and service quality.

## Dataset

The dataset contains information on passenger demographics, flight details, and feedback. Key features include:

- Demographic Information: Age, Gender, etc.

- Flight Details: Flight Distance, Travel Class.

- Service Ratings: Inflight Entertainment, Seat Comfort, Food Quality, etc.

- Target Variable: Satisfaction level (Satisfied/Dissatisfied).

### Data Preprocessing

- Handling missing values.

- Encoding categorical features.

- Scaling numerical features.

- Splitting data into training and testing sets.

## Models Used
 1. Naive Bayes
 2. Random Forest Classifier
 3. Decision Tree Classifier 

## Installation

### Prerequisites

Ensure you have Python (>=3.8) and the following libraries installed:

- pandas

- numpy

- matplotlib

- seaborn

- scikit-learn

- xgboost

## Project Workflow

### Exploratory Data Analysis (EDA):

- Visualize relationships between features.

- Identify trends in passenger satisfaction.

### Data Preprocessing:

- Handle missing values, encode categorical features, and scale numeric data.

### Model Training:

- Train multiple machine learning models such as Logistic Regression, Random Forest, and XGBoost.

- Perform hyperparameter tuning using GridSearchCV or RandomizedSearchCV.

### Evaluation:

- Use metrics like Accuracy, Precision, Recall, F1-Score, and ROC-AUC to assess model performance.

- Visualize model performance using confusion matrices, ROC curves, and feature importance plots.

## Technologies Used

- Programming Language: Python

- Libraries: pandas, numpy, matplotlib, seaborn, scikit-learn, xgboost

- Tools: Jupyter Notebook, GitHub

## Results

### 1. Naive Bayes:

- Accuracy: 81%

- Precision: 84%

- Recall: 86%

- ROC-AUC Score: 89%

### 2. Random Forest Classifier:

- Accuracy: 98%

- Precision: 99%

- Recall: 99%

- ROC-AUC Score: 99%

### 3. Decision Tree Classifier:

- Accuracy: 75%

- Precision: 76%

- Recall: 75%

- ROC-AUC Score: 81%
