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
1. Download the Cats and Dogs Dataset from Kaggle and Unzip the dataset. (https://www.kaggle.com/c/dogs-vs-cats/data)

**Step 2: Run Model A Jupyter Notebook**
1. Open ModelA.ipynb in Jupyter Notebook.
2. Run all cells to train the initial classification model on the dataset.
Output: This model provides a baseline classification model and saves initial predictions for further processing.

**Step 3: Run ECGR8119_Midterm_SRGAN_VGG Notebook**
1. Open ECGR8119_Midterm_SRGAN_VGG.ipynb in Jupyter Notebook.
This notebook uses an SRGAN (Super-Resolution GAN) to enhance the resolution of images in the dataset.
2. Run all cells in this notebook to:
3. Train the SRGAN model.
4. Apply the SRGAN model to enhance image resolution.
5. Use a VGG-based network for feature extraction.
Output: Enhanced images and extracted features, which will serve as input to Model B for improved classification.

**Step 4: Run Model B Jupyter Notebook**
1. Open ModelB.ipynb in Jupyter Notebook.
2. Run all cells to train the final classification model using the enhanced dataset and extracted features from the SRGAN model.
Output: Final classification model with improved performance metrics.

**valuation and Metrics**
Each model notebook includes code for evaluating performance using metrics such as Precision, Recall, F1 Score, and AUC. See each notebook for detailed instructions on generating these metrics.

**Additional Notes**
1. Ensure each notebook is run in sequence, as Model B depends on outputs from the previous steps.
2. Use TensorBoard or other visualization tools to monitor training performance across notebooks.

