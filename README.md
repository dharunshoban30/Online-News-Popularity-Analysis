# 🗞️ Online News Popularity Analysis using Data Mining and Machine Learning Techniques

## Overview
This project explores the performance of different machine learning models using two feature selection techniques: Linear Discriminant Analysis (LDA) and Principal Component Analysis (PCA) to predict the number of shares a social media post will receive using the Online News Popularity dataset from [UCI Machine Learning Repository](https://archive.ics.uci.edu/). 

## Objectives
The primary objectives of this project are:
  - Build five machine learning models capable of classifying online post shares.
  - Apply two distinct feature selection algorithms.
  - Compare the performance of different models.

## Description
1. **Data Preprocessing:**
    - Loading the dataset and handling missing values, duplicate rows, discretization, normalization, and SMOTE.
      
3. **Feature Selection:**
    - Applying LDA and PCA to reduce the dimensionality of the dataset.
    - Comparing the effects of both techniques on model performance.

5. **Model Training and Evaluation:**
    - Implementing five machine learning models, including:
      - Random Forest
      - Decision Tree
      - K-Nearest Neighbors (KNN)
      - CatBoost
      - XGBoost
    - Evaluating each model using the following metrics:
      - accuracy
      - precision
      - recall
      - F1-score
      - ROC-AUC

7. **Visualization:**
    - Visualizing the ROC-AUC scores of each model with LDA and PCA using line plots for easy comparison.

## Output
Results revealed that PCA outperformed LDA, yielding higher model performance across all metrics. Notably, Random Forest and Extreme Gradient Boosting emerged as the top-performing models, achieving 93% accuracy, precision, recall, and F1-score, with an ROC AUC of 99%. Overall, this study underscores the efficacy of PCA in feature selection and highlights the superiority of Random Forest and Extreme Gradient Boosting in predicting social media post shares.

## References
  - Dataset: [Online News Popularity Dataset](https://archive.ics.uci.edu/dataset/332/online+news+popularity)
