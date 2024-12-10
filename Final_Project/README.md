# ECGR-8119-Applied-AI Final Project 
# Leveraging Predictive Models for Workload Execution Prediction and Efficient Resource Allocation

This project focuses on predicting task execution time in Google cluster workloads using machine learning models and optimizing resource allocations (CPU and memory) to ensure efficient usage. The workflow includes data preprocessing, model training, and implementing an optimization framework leveraging predictive insights for workload management.

**Repository File Structure & Contents**

1. ECGR8119_FinalProject_Part1.ipynb - A Jupyter Notebook used to train three Models - Random Forest, Decision Tree, and Neural Network. The notebook contains Training Results and Plots. 
2. ECGR8119_FinalProject_Part2.ipynb - A Jupyter Notebook used to implement the Optimization Framework and plot the results.

**Project Overview**

Dataset --> Google Cluster Trace Dataset from Kaggle.
Models Used:
1. Model 1: Random Forest - Powerful ensemble learning model that combines multiple decision trees to improve prediction accuracy. The model was trained on 80% of the data, using 100 estimators to reduce overfitting and enhance generalization.

2. Model 2: Decision Tree - Simple yet effective model used for classification and regression. The decision tree model was trained to predict the execution time of a particular task, and its performance was evaluated by tuning parameters such as max depth.

3. Neural Network - A multi-layer perceptron (MLP) neural network was implemented and trained for 10, 25, 50 and 100 epochs to assess performance and convergence. The network architecture included an input layer, 2 hidden layers, and an output layer for regression.


**Steps to Replicate the Project:** 

**Step 1: Download Dataset**
1. Download the Google Cluster Trace Dataset from Kaggle and Unzip the dataset. (https://www.kaggle.com/datasets/derrickmwiti/google-2019-cluster-sample)

**Step 2: Run Part 1 Jupyter Notebook**
1. Open ECGR8119_FinalProject_Part1.ipynb in Jupyter Notebook.
2. Run initial cells to preprocess, normalize and visualize the processed data. Once completed, save the preprocessed PANDAS DATAFRAME for easier access.
3. Train the preprocessed data on Model 1 - Random Forest. Print and Plot the results.
4. Next, train the same data on Decision Tree. Print and Plot the results.
5. Finally, train the data on Neural Network. Plot the results. 
Output: Trained models (Random Forest, Decision Tree, and Neural Network)

**Step 3: Run Part 2 Jupyter Notebook**
1. 

**valuation and Metrics**
Each model notebook includes code for evaluating performance using metrics such as Mean Absolute Error and R-squared. See each notebook for detailed instructions on generating these metrics.

**Additional Notes**
1. Ensure each notebook is run in sequence, as Part 2 depends on outputs from the previous steps.

