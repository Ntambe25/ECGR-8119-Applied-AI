# ECGR-8119-Applied-AI Final Project 
# Leveraging Predictive Models for Workload Execution Prediction and Efficient Resource Allocation

This project explores the application of deep learning techniques to improve image classification accuracy through image resolution enhancement and data augmentation. In Part A, a baseline is established by training a MobileNetV2 model on a dataset of cats and dogs, where each image is downscaled to 128x128 pixels. This initial phase allowed to evaluate the model’s performance on lower-resolution images, setting a standard for comparison.

In Part B, super-resolution techniques were introduced to augment the dataset. A Super-Resolution Generative Adversarial Network (SRGAN) was trained on 32x32 images for 160 epochs to generate higher-quality, 128x128 images. Once trained, the SRGAN model was used to generate approximately 2000 super-resolved images, which were then added to the original cats and dogs dataset. The expanded dataset, containing both original and super-resolved images, was then used to re-train the MobileNetV2 model.

By comparing the performance metrics of the MobileNetV2 model trained on the original dataset to the model trained on the enhanced dataset, the aim was to assess the impact of super-resolution and data augmentation on classification accuracy.

This project, thus investigated whether super-resolution can improve model performance on low-resolution image datasets, with implications for AI applications in resource-constrained environments where high-resolution data may be scarce.

**Repository File Structure & Contents**

1. ECGR8119_Midterm_ModelA.ipynb - A Jupyter Notebook used to train Model A and print the Results
2. ECGR8119_Midterm_SRGAN_VGG.ipynb - A Jupyter Notebook used to train SRGAN Model on downscaled 32x32 cats and dogs images (Since, the File size was around 62 MB, the File was pushed to Github using Git Large File Storage (LFS), and its contents may not be displayed in Github)
3. ECGR8119_Midterm_ModelB.ipynb - A Jupyter Notebook used to train Model B and print the Results

**Project Overview**

Dataset --> Cats and Dogs Dataset from Kaggle.
Models Used:
1. Model A: Baseline classification model.
2. ECGR8119_Midterm_SRGAN_VGG --> An SRGAN (Super-Resolution GAN) model to enhance image resolution, followed by a VGG-based model for feature extraction.
3. Model B: A refined classifier leveraging enhanced images from the SRGAN model for improved accuracy.

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

