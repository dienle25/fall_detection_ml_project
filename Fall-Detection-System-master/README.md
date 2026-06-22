# 🏃‍♂️ Fall Detection System: Machine Learning & Real-time Vision

**Course Project: Machine Learning**

## 1. Introduction
Fall detection among the elderly is a crucial issue in healthcare due to the high risk of injury and related complications. Falls are a leading cause of hospitalizations and long-term disabilities in older adults, making early and accurate detection vital for timely intervention. 

This project presents a **hybrid approach** to fall detection. It features a rigorous academic evaluation of various Machine Learning models using kinematic sensor data (SisFall Dataset) and is complemented by a non-intrusive, real-time Computer Vision demonstration (YOLO + MediaPipe) for camera-based monitoring.

## 2. Core Dataset: SisFall Overview
The core ML models are trained and evaluated on the **SisFall dataset**, which focuses on elderly individuals prone to falls.
- **Sensor Data:** Raw time-series data from accelerometers and gyroscopes worn at the waist.
- **Falls and Activities:** Includes simulated forward, backward, and sideways falls, alongside normal Activities of Daily Living (ADL) like walking, sitting, standing, and bending.
- **Subjects:** Data from both young and elderly participants, ensuring diverse movement characteristics for better model generalization.
- **Access the Dataset:** 
  - [SisFall Dataset](https://drive.google.com/uc?id=1-E-TLd5_J-DDWZXkuYL-moMpoezlMn4Z)
  - [SisFall Enhanced Dataset (Labels)](https://drive.google.com/uc?id=1gvOuxPc8dNgTnxuvPcVuCKifOf98-TV0)

## 3. Project Architecture & Implementation Details

The repository is structured into two main components: the Academic ML Core and the Real-time Vision Demo.

### Part A: Machine Learning Core (Sensor-based)
*   **`data_processor.py`**: Processes the SisFall dataset and converts it into a time-series format (`X_train`, `y_train`, `X_test`, `y_test`).
*   **`traditional_models.py`**: Implementation and evaluation of traditional machine learning models (SVM, KNN, Random Forest, Decision Tree).
*   **`deep_models.py`**: Implementation and evaluation of deep learning models (CNN, LSTM).
*   **`utils.py`**: Collection of utility functions used throughout the data processing pipeline.
*   **`Main.ipynb`**: The main Jupyter Notebook for training models and generating detailed evaluation data frames.

### Part B: Real-time Vision Demo (Camera-based)
Located in the `Realtime_Vision_Demo/` directory, this module translates the concept of fall detection into a camera-based system without requiring wearable sensors.
*   **Technologies:** YOLO (Object/Person Detection) + MediaPipe (Pose/Landmark Estimation) + Random Forest Classifier.

## 4. Evaluation Results
The models were evaluated based on standard classification metrics. Below is a summary of the performance on the test set:

| Model | Accuracy | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- | :--- |
| KNN | 91.2% | 90.5% | 91.8% | 91.1% |
| Decision Tree | 89.5% | 88.0% | 89.2% | 88.6% |
| **Random Forest** | **96.8%** | **96.5%** | **97.0%** | **96.7%** |
| CNN | 95.5% | 95.0% | 96.1% | 95.5% |
| LSTM | 94.2% | 93.8% | 94.5% | 94.1% |
*(Detailed performance reports and confusion matrices are available in the notebook outputs).*

## 5. How to Run

### Running the Machine Learning Models (SisFall)
1. Install the required core libraries:
```bash
   pip install -r requirements.txt
