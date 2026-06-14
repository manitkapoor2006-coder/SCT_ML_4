🖐️ Hand Gesture Recognition System
A production-ready Machine Learning project that classifies hand gestures from images using a Random Forest classifier trained on the LeapGestRecog dataset. Built with Python · Scikit-learn · Streamlit · Matplotlib · Seaborn.

📋 Table of Contents
Project Overview
Dataset
Image Preprocessing
Model Architecture
Evaluation Metrics
Project Structure
Installation & Running
Screenshots
Tech Stack
Project Overview
Hand Gesture Recognition (HGR) is a computer-vision task that maps hand poses captured in images to semantic labels (e.g. "Thumbs Up", "Fist", "OK"). This project implements a full ML pipeline:

Raw Image → Greyscale → Resize → Flatten → Standardise → Random Forest → Predicted Gesture
Applications include sign-language interpretation, touchless UI control, gaming, and robotics.

Dataset
Property	Value
Name	LeapGestRecog
Source	Kaggle — gti-upm/leapgestrecog
Sensor	Leap Motion Controller (near-infrared)
Subjects	10 participants
Classes	10 hand gestures
Format	Greyscale PNG
Licence	CC0 (Public Domain)
Gesture Classes
ID	Folder	Label
01	01_palm	Palm
02	02_l	L Shape
03	03_fist	Fist
04	04_fist_moved	Fist (Moved)
05	05_thumb	Thumb Up
06	06_index	Index Finger
07	07_ok	OK Sign
08	08_palm_moved	Palm (Moved)
09	09_c	C Shape
10	10_down	Down
Image Preprocessing
Each image goes through a deterministic preprocessing pipeline before entering the model:

Load — PIL.Image.open() reads PNG/JPG images.
Greyscale — .convert("L") converts RGB to a single luminance channel.
Resize — .resize((64, 64)) standardizes image dimensions.
Flatten — numpy.flatten() transforms the image into a 1-D feature vector.
Standardise — StandardScaler scales features to zero mean and unit variance.
Model Architecture
Algorithm — Random Forest Classifier
Hyperparameter	Value
n_estimators	200
max_depth	None
min_samples_split	2
class_weight	balanced
random_state	42
n_jobs	-1
Why Random Forest?
Ensemble learning with multiple decision trees.
Handles high-dimensional image features efficiently.
Provides probability scores using predict_proba.
Robust against overfitting compared to a single decision tree.
Works well with structured image feature vectors.
Train/Test Split
80% Training Data
20% Testing Data
stratify=y used to preserve class distribution.
Evaluation Metrics
Metric	Formula	Meaning
Accuracy	Correct / Total	Overall correctness
Precision	TP / (TP + FP)	Positive prediction quality
Recall	TP / (TP + FN)	Detection capability
F1 Score	2 × P × R / (P + R)	Balance between Precision & Recall
Confusion Matrix	True vs Predicted	Error distribution analysis
Weighted averages are used for multi-class evaluation.

Project Structure
hand_gesture_recognition/
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
│
├── models/
│   ├── gesture_model.pkl
│   ├── label_encoder.pkl
│   ├── scaler.pkl
│   ├── metrics.json
│   └── confusion_matrix.png
│
├── utils/
│   ├── __init__.py
│   └── predict.py
│
├── sample_images/
└── dataset/
Installation & Running
Prerequisites
Python 3.9+
pip
Step 1 — Clone Repository
git clone https://github.com/your-username/hand-gesture-recognition.git
cd hand-gesture-recognition
Step 2 — Install Dependencies
pip install -r requirements.txt
Step 3 — Train Model
python train_model.py
Step 4 — Launch Streamlit App
streamlit run app.py
Open:

http://localhost:8501
Streamlit Dashboard Pages
Page	Description
🏠 Dashboard	Project overview and KPIs
📊 Model Analytics	Metrics and confusion matrix
🔮 Predict Gesture	Upload image and predict gesture
🖼️ Dataset Preview	Sample images and distributions
ℹ️ About	Project details and workflow
Tech Stack
Library	Purpose
streamlit	Web application
scikit-learn	Machine Learning
numpy	Numerical computations
pandas	Data processing
Pillow	Image preprocessing
opencv-python-headless	Computer vision utilities
matplotlib	Visualization
seaborn	Heatmaps
joblib	Model persistence
kaggle	Dataset download
tqdm	Progress tracking
Acknowledgements
LeapGestRecog Dataset by GTI — Universidad Politécnica de Madrid.
Built as part of a Machine Learning Internship Project.
Demonstrates data preprocessing, model training, evaluation, and deployment.
