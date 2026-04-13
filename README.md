# Face Emotion Classification with Neural Networks

A from-scratch implementation of a single hidden layer neural network in Python for classifying face emotion data, trained with stochastic gradient descent and backpropagation.

## Overview

This project implements a feedforward neural network without any deep learning libraries — just NumPy. It classifies facial emotion data from a `.mat` dataset and includes a separate synthetic data example demonstrating the network on two non-linear classification problems.

## Features

- **Single hidden layer neural network** built entirely with NumPy
- **Sigmoid activation** for both hidden and output layers
- **Stochastic gradient descent** with configurable learning rate and epochs
- **8-fold cross-validation** for evaluating generalization performance
- **Synthetic data demo** showing the network learning non-linear decision boundaries

## Dependencies

- NumPy
- Matplotlib
- SciPy (`loadmat` for reading `.mat` files)

## Dataset

`face_emotion_data.mat` containing a feature matrix `X` and binary labels `y` for 128 face samples.

## Results

### Face Emotion Classification

- Achieves 0% training error on the full dataset
- ~96% test accuracy via 8-fold cross-validation (varies slightly due to SGD randomness)

### Synthetic Data Demo

Trains on 10,000 generated samples across two non-linear classification tasks (circular boundary and cubic boundary), demonstrating the network's ability to learn complex decision surfaces.

## Hyperparameters

| Parameter | Value |
|---|---|
| Learning rate (`a`) | 0.05 |
| Hidden neurons (`M`) | 32 |
| Epochs (`L`) | 200 |

## Structure

| File | Description |
|---|---|
| `face_emotion.ipynb` | Main notebook — training, cross-validation, and plots |
| `face_emotion_data.mat` | Face emotion dataset |
