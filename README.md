# K-Nearest Neighbors (KNN) Classification Project

## 📄 Dataset Description
* **Source:** [breast cancer wisconsin dataset]
* **Objective:** Predict [Target_Label] based on [Number] input features.
* **Features:** key features (mean radius	, mean texture,	mean perimeter	,mean area	,mean smoothness,	mean compactness,	mean concavity,	mean concave points	,mean symmetry,	mean fractal dimension,	...	worst perimeter,	worst area,	worst smoothness	,worst compactness,	worst concavity	,worst concave points,	worst symmetry	,worst fractal dimension	,Target,	Target_Label,)




3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

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
* **F1-Score (0.98 & 0.99):** The harmonic mean of precision and recall is exceptionally high for both classes, proving that the model is well-balanced and not biased toward the majority class (Class 1, with 72 samples) despite the slight class imbalance.

 Conclusion
The systematic hyperparameter tuning using `GridSearchCV` paired with feature standardization successfully eliminated the bias typically found in vanilla KNN implementations. The model generalizes beautifully to unseen data.
