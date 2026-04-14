# Face Emotion Classification

## Overview
This project builds a complete machine learning pipeline from scratch, progressively improving model performance for face emotion classification using increasingly advanced techniques.

Starting from a simple linear model, the project evolves through dimensionality reduction, kernel methods, and nonlinear modeling to capture complex patterns in data.

---

## Pipeline Architecture

### 1. Linear Classification (Baseline)
- Implemented least squares classifier
- Evaluated performance using all features vs selected top features
- Used 8-fold cross validation for performance estimation

**Key Idea:** Establish a simple baseline model

---

### 2. Dimensionality Reduction (SVD)
- Applied Singular Value Decomposition (SVD)
- Constructed rank-1 and rank-2 approximations
- Analyzed reconstruction error and data structure

**Key Idea:** Reduce noise and uncover structure in data

---

### 3. Kernel Methods (Nonlinear Modeling)
- Built Gaussian kernel classifier
- Implemented kernel ridge regression
- Tuned hyperparameters (σ, λ) using cross-validation
- Visualized nonlinear decision boundaries

**Key Idea:** Transform linear models into nonlinear ones using kernels

---

### 4. Advanced Modeling (Neural Networks)
- Implemented multi-layer neural network (if included)
- Trained using gradient-based optimization
- Compared performance with previous models

**Key Idea:** Learn features automatically from data

---

## Features

- End-to-end ML pipeline from scratch
- No high-level ML libraries (manual implementations)
- Cross-validation and hyperparameter tuning
- Visualization of decision boundaries and model behavior
- Both classification and regression tasks

---

## Results & Insights

- Linear models provide fast but limited performance  
- SVD reveals low-dimensional structure in data  
- Kernel methods achieve near-perfect training accuracy  
- Cross-validation prevents overfitting  
- Nonlinear models significantly improve classification performance  

---

## Technologies Used

- Python
- NumPy
- SciPy
- Matplotlib
