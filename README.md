
# Cross-Sell Prediction Model 
 
## Overview 
This repository contains code for a machine learning project that predicts cross-selling opportunities using various classifiers and techniques. 
 
## Code Structure 
-  main.py : Main script containing the machine learning pipeline. 
-  train.csv : Dataset used for training the models. 
-  README.md : You are here! 
 
## Getting Started 
To run the code, follow these steps: 
1. Clone the repository:

git clone https://github.com/yourusername/cross-sell-prediction.git
2. Install the required libraries:
bash
pip install pandas numpy scikit-learn matplotlib seaborn imbalanced-learn
3. Run the  main.py  script:

python main.py
## Data Preprocessing 
The  data_prep  function preprocesses the dataset by: 
- Dropping unnecessary columns. 
- One-hot encoding categorical variables. 
- Binning numerical features. 
- Handling imbalanced data. 
 
## Model Training 
Three classifiers are trained and evaluated: 
- Logistic Regression 
- Decision Tree 
- Random Forest 
 
## Hyperparameter Tuning 
GridSearchCV is used to tune the Random Forest model for improved performance. 
 
## Results 
The models are evaluated based on accuracy and F1 scores on both training and testing sets. 
 

