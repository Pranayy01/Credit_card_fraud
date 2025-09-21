# Credit Card Fraud Detection

A machine learning project to detect fraudulent credit card transactions using multiple algorithms and probability-based evaluation metrics.

## 📊 Dataset
- **Source**: [Kaggle Credit Card Fraud Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)  
- ~284,000 transactions with **0.17% fraud cases** (highly imbalanced).

## ⚙️ Approach
1. **Exploratory Data Analysis (EDA)**
   - Inspected data distribution and identified strong class imbalance.
   - Scaled `Amount` and `Time` features using `RobustScaler` and `MinMaxScaler`.

2. **Class Imbalance Handling**
   - Created a **balanced dataset** (fraud vs non-fraud equalized) to benchmark models.  
   - Models were trained on both **imbalanced** (realistic scenario) and **balanced** datasets.  

3. **Models Trained**
   - Logistic Regression  
   - Random Forest Classifier  
   - Gradient Boosting Classifier  
   - Support Vector Machine (LinearSVC)  
   - Neural Network (Keras Sequential)  

4. **Evaluation Metrics**
   - Accuracy, Balanced Accuracy  
   - Precision, Recall, F1 (fraud class focus)  
   - ROC-AUC, PR-AUC  
   - Confusion Matrix  
   - Log-loss & Brier Score for calibration  

## 📈 Results
- On **balanced dataset**:  
  - All models achieved strong fraud detection performance.  
  - Demonstrated effectiveness of resampling in highly imbalanced problems.  

- On **imbalanced dataset**:  
  - Accuracy was very high but misleading due to dominance of non-fraud cases.  
  - Neural Network outperformed others by maintaining high fraud recall while minimizing false positives.  

## ✅ Final Model Selection
- **Chosen Model**: **Neural Network (Keras)** trained on the **imbalanced dataset**  
- **Reasoning**:  
  - Training on the imbalanced dataset preserves the **real-world distribution of fraud** (fraudulent transactions are rare).  
  - In production, models must handle this imbalance directly instead of relying solely on resampled data.  
  - **Recall was prioritized**: approving fraudulent transactions is more costly than false alarms.  

### 🔹 Final Model Metrics (Neural Network on Imbalanced Dataset)
- **Accuracy**: 99.93%  
- **Balanced Accuracy**: 91.09%  
- **Precision (fraud)**: 82.2%  
- **Recall (fraud)**: 82.2%  
- **F1 (fraud)**: 82.2%  
- **MCC**: 0.82  
- **ROC-AUC**: 0.982  
- **PR-AUC**: 0.804  
- **Log-loss**: 0.0049  
- **Brier Score**: 0.0006  

## 🔑 Key Highlights
- **Business alignment**: Prioritized **Recall** over Accuracy, aligning with real-world risk (false negatives = undetected fraud).  
- **Balanced vs Imbalanced training**: Compared both approaches, showing awareness of real ML challenges.  
- **Systematic model comparison**: Evaluated multiple algorithms rather than relying on a single one.  
- **Calibrated probabilities**: Used Log-loss and Brier Score to ensure predicted fraud probabilities are reliable.  

## 🔮 Future Enhancements
- Hyperparameter tuning with GridSearch/RandomizedSearch.  
- Threshold optimization to push recall higher while managing precision.  
- Deploying as a REST API or Streamlit dashboard for real-time fraud detection.  
