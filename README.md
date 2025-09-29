# README.md — Machine Learning Coursework and Labs (COMP4139)

## Module Overview

This repository contains coursework, lab notebooks, and reports from the COMP4139 / COMP3009 module titled Machine Learning at the University of Nottingham.

The module introduces core concepts, tools, and techniques in modern machine learning, including:

- Data preprocessing and representation  
- Feature selection and dimensionality reduction  
- Supervised learning (linear and nonlinear models)  
- Evaluation metrics and model selection  
- Deep learning models (ANN, CNN, RNN, Transformers)  
- Unsupervised learning (clustering, PCA, DBSCAN)  
- Generative models (Autoencoders, GANs)  
- Reinforcement learning fundamentals (Q-learning, policy gradients)

## Lab Sessions

The lab notebooks are hands-on exercises covering the end-to-end ML pipeline using Python on Google Colab.

| Lab    | Topic                         | Key Concepts                                      |
|--------|-------------------------------|--------------------------------------------------|
| Lab 1  | Python Basics                 | Loops, functions, NumPy, Pandas                  |
| Lab 2  | Data Preprocessing            | Scaling, encoding, cleaning, outliers            |
| Lab 3  | ML Models                     | Linear/Logistic Regression, Decision Trees       |
| Lab 4  | Advanced Models               | SVM, ANN, Random Forests                         |
| Lab 5  | Evaluation                    | Accuracy, Precision, Recall, F1, AUC             |
| Lab 6  | Assignment 1 Submission Prep | Feature selection and model validation           |
| Lab 7  | TensorFlow and CNN Basics    | Layers, activations, CNN implementation          |

Key libraries used: scikit-learn, pandas, matplotlib, tensorflow, numpy.

## Assignment: Predicting Breast Cancer Outcomes

Files:
- ML_REPORT.pdf  
- COMP4139_Assignment1.ipynb

### Task

This project applies machine learning to predict two breast cancer outcomes:

- PCR (Pathological Complete Response) – Classification  
- RFS (Recurrence-Free Survival) – Regression

### Dataset

Based on the I-SPY2 Clinical Trial, the dataset includes:

- 11 clinical features (e.g. age, receptor status, tumor stage)  
- 107 MRI-derived features extracted via PyRadiomics  
- 400 unique patient records  

### Feature Selection

Several techniques were explored to reduce dimensionality and improve model performance:

- Correlation Filtering  
- Tree-Based Filtering (e.g. Random Forest feature importance)  
- LASSO Regularization (L1 penalty)  
- Principal Component Analysis (PCA)  

### Models Used

Classification:
- Decision Tree Classifier  
- Random Forest Classifier (Best for PCR)  
- Gradient Boosting  
- MLP Classifier  

Regression:
- Linear Regression  
- Lasso Regression  
- Support Vector Regression (SVR)  
- MLP Regressor  
- Random Forest Regressor (Best for RFS)  

### Evaluation

- Classification: Balanced Accuracy, F1 Score, Confusion Matrix  
- Regression: Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), R² Score  
- 5-fold cross-validation was used for evaluation and robustness  

### Key Results

| Task                | Best Model       | Metric Value             |
|---------------------|------------------|---------------------------|
| Classification (PCR)| Random Forest    | 84.19% Accuracy           |
| Regression (RFS)    | Random Forest    | Lowest MAE, Highest R²    |

## Contributors

- Lukshan Sharvaswaran  
- Arunavo Dutta  
- Shubhankar Shahade  
- Yalamanchili Prashanth  
- Veerendra Kumar Dangeti  
