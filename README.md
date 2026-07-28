# Vibration-Based Motor Fault Classifier

A mechanical engineering and machine-learning project that uses vibration data to distinguish between healthy operation and a loose bearing block on a rotating motor test rig.

## Project Overview

This project combines mechanical design, sensor instrumentation, signal processing, and machine learning.

I designed and assembled a rotating machinery test rig using a DC motor, shaft, bearings, custom CAD-designed mounts, an Arduino Uno, and an MPU6500 inertial measurement unit.

Vibration data was collected under two operating conditions:

- Healthy
- Loose bearing block

The raw acceleration signals were divided into one-second windows and converted into time- and frequency-domain features. A Random Forest classifier was then evaluated using grouped cross-validation.

## Current Results

The pilot dataset contains:

- 20 independent motor recordings
- 10 healthy recordings
- 10 loose-bearing recordings
- 160 one-second vibration windows
- Approximately 250 Hz sampling rate
- One operating speed: PWM 25%

The pilot Random Forest achieved:

- **81.25% held-out window accuracy**
- **0.857 ROC AUC**
- **81.25% loose-bearing precision**
- **81.25% loose-bearing recall**
- **81.25% F1 score**
- **81.25% healthy specificity**

Grouped five-fold cross-validation was used so that windows from the same motor recording could not appear in both the training and testing sets.

> These results are preliminary. The dataset is small and currently represents only one motor speed, one rig configuration, and one fault type.

## Hardware

- 24 V RS-775 DC motor
- Shaft and flexible coupling
- Two bearing blocks
- Custom CAD-designed motor and bearing mounts
- Arduino Uno
- MPU6500 accelerometer and gyroscope
- PWM motor-speed controller
- Switching power supply
- 3D-printed components
- Wooden baseplate

## Software and Tools

- Python
- Arduino C/C++
- scikit-learn
- pandas
- NumPy
- Matplotlib
- Fusion 360
- Arduino IDE
- VS Code

## Machine-Learning Pipeline

The current workflow is:

1. Collect acceleration data from the MPU6500.
2. Save each motor run as a CSV file.
3. Divide each recording into one-second windows.
4. Extract time-domain and frequency-domain features.
5. Validate the feature dataset.
6. Create recording-grouped cross-validation splits.
7. Train a Random Forest classifier.
8. Generate held-out window predictions.
9. Evaluate accuracy, precision, recall, F1, specificity, and ROC AUC.
10. Aggregate window probabilities into recording-level predictions.

## Extracted Features

Time-domain features include:

- Mean
- Standard deviation
- Root mean square acceleration
- Minimum and maximum
- Peak-to-peak amplitude
- Skewness
- Kurtosis
- Crest factor

Frequency-domain features include:

- Dominant frequency
- Spectral centroid
- Spectral rolloff
- Total spectral power
- Frequency-band power

Features were calculated for individual acceleration axes and combined vibration magnitude.

## Repository Structure

```text
motor-condition-monitoring-test-rig/
│
├── data/
│   └── Extracted feature datasets
│
├── scripts/
│   ├── 01_validate_features.py
│   ├── 02_prepare_ml_data.py
│   ├── 03_validate_grouped_splits.py
│   ├── 04_train_grouped_random_forest.py
│   ├── 05_evaluate_window_predictions.py
│   └── 06_evaluate_recording_predictions.py
│
├── models/
│   └── Saved Random Forest fold models
│
├── outputs/
│   ├── Validation results
│   ├── Held-out predictions
│   ├── Evaluation metrics
│   ├── Confusion matrices
│   └── ROC curves
│
└── README.md
