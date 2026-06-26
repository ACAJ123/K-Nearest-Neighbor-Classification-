# K-Nearest Neighbors (KNN) Classification Project

## Project Overview

This project demonstrates the implementation of a K-Nearest Neighbors (KNN) Classification model using Python and Scikit-learn. The objective is to classify breast cancer diagnoses as benign or malignant using machine learning techniques.

The project covers the complete machine learning workflow, including data preprocessing, feature scaling, train-test splitting, hyperparameter tuning, model training, evaluation, and interpretation of results.

---

## Objectives

* Load and explore the Breast Cancer dataset.
* Prepare and preprocess data for machine learning.
* Implement a K-Nearest Neighbors (KNN) classifier.
* Optimize model performance using hyperparameter tuning.
* Evaluate model performance using multiple classification metrics.
* Interpret results to assess model effectiveness.

---

## Dataset

The project uses the Breast Cancer Wisconsin dataset available through Scikit-learn.

### Features

The dataset contains several numerical features computed from digitized images of breast mass cell nuclei, including:

* Mean Radius
* Mean Texture
* Mean Perimeter
* Mean Area
* Mean Smoothness
* And other diagnostic measurements

### Target Variable

* 0 = Malignant
* 1 = Benign

---

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn

---

## Project Workflow

### 1. Data Loading

The dataset is loaded and converted into a Pandas DataFrame for analysis.

### 2. Data Preprocessing

The following preprocessing steps were applied:

* Feature and target separation
* Train-test split
* Feature standardization using StandardScaler

### 3. Model Development

A K-Nearest Neighbors classifier was implemented using Scikit-learn.

### 4. Hyperparameter Tuning

GridSearchCV and cross-validation were used to identify the optimal value of k (number of neighbors) and improve model performance.

### 5. Model Evaluation

The model was evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix
* Classification Report

### 6. Visualization

Visualizations were created to:

* Analyze cross-validation performance across different k values
* Display confusion matrix results
* Support interpretation of model performance

---

## Results

The optimized KNN model achieved strong classification performance on the test dataset.

Evaluation metrics demonstrated the model's ability to accurately distinguish between malignant and benign tumors while minimizing classification errors.

The confusion matrix showed a low number of false positives and false negatives, indicating reliable predictive performance.

---

## Key Findings

* Feature scaling significantly improved KNN performance.
* Hyperparameter tuning helped identify the most effective k value.
* Cross-validation improved model reliability and reduced overfitting risk.
* The final model demonstrated strong predictive capability for breast cancer classification.

---

## How to Run

### Install Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### Run the Notebook

Open Jupyter Notebook and execute all cells in sequence.

```bash
jupyter notebook
```

---

## Project Structure

```text
project/
│
├── KNN_Classification.ipynb
├── README.md
└── requirements.txt
```

---

## Future Improvements

* Compare KNN with Logistic Regression and Random Forest models.
* Perform feature selection to improve efficiency.
* Explore advanced hyperparameter optimization techniques.
* Deploy the model as a web application.

---

## Author

Clinton Ayomide Aleshinloye

This project was completed as part of a Machine Learning classification exercise focused on applying K-Nearest Neighbors (KNN) for predictive analytics.
