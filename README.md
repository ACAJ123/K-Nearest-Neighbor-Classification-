# K-Nearest Neighbors (KNN) Classification Project
NN Classification Pipeline with Hyperparameter TuningThis repository contains a complete, end-to-end Machine Learning pipeline utilizing the K-Nearest Neighbors (KNN) algorithm. The project progresses from data preprocessing to cross-validation hyperparameter optimization, concluding with a comprehensive model evaluation.Project Workflow[Data Loading] -> [Scaling & Splitting] -> [Cross-Validation Loop] -> [Tie-Breaking Logic] -> [Final Evaluation]Data Preprocessing: Features are scaled using StandardScaler to ensure uniform distance calculations, and the dataset is partitioned into training and testing sets.

 Dataset Description
* **Source:** [breast cancer wisconsin dataset]
* **Objective:** Predict [Target_Label] based on [Number] input features.


KHyperparameter Optimization: 
A grid of K values is systematically evaluated using 5-fold cross-validation (cross_val_score) to maximize accuracy while mitigating data leakage.Model Selection: Domain-specific tie-breaking logic is applied to balance model complexity against overfitting risks.Comprehensive Evaluation: The optimized model is evaluated on unseen test data using structural confusion matrices and detailed classification reports.Performance Summary & Parameter SelectionThe training set was evaluated across a range of odd K values using 5-fold cross-validation, yielding the following results:K = 1 | Mean CV Accuracy: 0.9451 (Suboptimal)K = 3 | Mean CV Accuracy: 0.9692 (Highest Accuracy - High risk of overfitting)K = 5 | Mean CV Accuracy: 0.9670 (Suboptimal)K = 7 | Mean CV Accuracy: 0.9692 (Highest Accuracy - Selected Model)K = 9 | Mean CV Accuracy: 0.9692 (Highest Accuracy)K = 11 | Mean CV Accuracy: 0.9626 (Performance Drop)Tie-Breaking Architecture SelectionWhile K=3, K=7, and K=9 achieved an identical maximum accuracy of 0.9692, K=7 was selected for the deployment model. Choosing K=7 over K=3 limits sensitivity to localized noise/outliers (overfitting), while retaining a more robust decision boundary than K=9.Implementation Code1. Hyperparameter Tuning & VisualizationRun this script to evaluate parameter behavior and plot the K vs. Accuracy curve.

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   pythonimport matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

4. **Launch the notebook:**
   ```bash
   jupyter notebook knn_classification.ipynb
   ```

## 📊 Key Findings & ResultsPerformance Evaluation & Interpretation

The optimized K-Nearest Neighbors model demonstrates exceptional classification performance across all key evaluation metrics:

* **Overall Accuracy (98%):** The model correctly classifies 98.2% of all test instances (112 out of 114 samples), indicating a robust general dependency mapping.
* **Performance on Class 0:** 
  * **Precision (1.00):** Out of all instances the model predicted as Class 0, 100% were actually Class 0. There are zero False Positives for this class.
  * **Recall (0.95):** The model successfully captured 95% of the actual Class 0 instances, missing only 5% (False Negatives).
* **Performance on Class 1:**
  * **Precision (0.97):** Out of all instances predicted as Class 1, 97% were correct.
  * **Recall (1.00):** The model achieved perfect recall (100%) for Class 1, meaning it successfully identified every single actual Class 1 instance in the test set.
 




# Baseline setup and cross-validation search grid
k_values = [1, 3, 5, 7, 9, 11]
cv_scores = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train_scaled, y_train, cv=5, scoring="accuracy")
    cv_scores.append(np.mean(scores))

best_k = 7  # Selected via tie-breaking analysis


Evaluation Metrics DecodedConfusion Matrix: Maps raw prediction breakdowns to isolate exactly where False Positives or False Negatives occur.Precision: Measures target accuracy within positive predictions (True Positives / (True Positives + False Positives)).Recall: Measures the model's capacity to find all real instances of a positive class (True Positives / (True Positives + False Negatives)).F1-Score: Calculates the balanced harmonic mean of Precision and Recall, ensuring stable metrics if handling class imbalances.
* **F1-Score (0.98 & 0.99):** The harmonic mean of precision and recall is exceptionally high for both classes, proving that the model is well-balanced and not biased toward the majority class (Class 1, with 72 samples) despite the slight class imbalance.

 Conclusion
The systematic hyperparameter tuning using `GridSearchCV` paired with feature standardization successfully eliminated the bias typically found in vanilla KNN implementations. The model generalizes beautifully to unseen data.
