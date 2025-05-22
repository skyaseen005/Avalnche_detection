# Avalnche_detection
Drone-Assisted Avalanche Search and Rescue System
This repository contains the implementation and research behind a drone-based vision system for assisting in avalanche search and rescue (SAR) operations. Leveraging UAVs equipped with computer vision and machine learning models, this system aims to reduce victim localization time and improve survival rates in post-avalanche scenarios.

**📌 Project Overview**
After an avalanche, time is critical for rescuing buried victims. Traditional methods like trained dogs or manual transceivers are often slow and resource-intensive. This project introduces a vision-based approach using UAVs (drones) to autonomously scan affected regions, identify potential victims using image processing and classification techniques, and alert rescue teams with precise GPS coordinates.

**🧠 Key Features**
UAV-based terrain surveillance using RGB and optionally thermal/IR cameras

Pre-processing pipeline to filter snow and identify potential victim regions

Feature extraction using:

Histogram of Oriented Gradients (HOG)

ORB (Oriented FAST and Rotated BRIEF)

Classification using Support Vector Machines (SVM)

Post-processing with Hidden Markov Models (HMM) for enhanced temporal consistency

Performance metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC

Operational workflow from drone deployment to alert generation

🔧 Technologies Used
Python

OpenCV

Scikit-learn

Matplotlib/Seaborn (for metrics and visualization)

UAV drone imagery dataset (custom/pre-recorded)

Classical ML classifiers: SVM, KNN

Edge detection & segmentation techniques

🛠 Methodology Summary
Preprocessing:

Grayscale conversion

Denoising, binarization

Histogram equalization

Canny edge detection

HSV thresholding for snow vs object separation

Feature Extraction:

HOG for shape-based features

ORB for keypoint-based binary descriptors

**Classification:**

SVM/KNN for distinguishing “victim” vs “no-victim” frames

HMM to utilize temporal context and reduce false alarms

Training:

Optimized using Adam optimizer

Augmented with flips, zoom, rotation, brightness variation

Evaluation:

Metrics: Accuracy, F1-Score, Confusion Matrix, ROC-AUC

**📈 Sample Results
Metric	Value
Precision	0.9208
Recall	0.7144
F1-Score	0.4896
**
Results based on a real-time UAV dataset captured in controlled environments.

**🔮 Future Scope**
Integration of thermal/infrared imaging

Real-time edge deployment on NVIDIA Jetson/Google Coral

Multi-modal data fusion (environmental + visual)

Temporal sequence analysis using 3D CNNs, ConvLSTM

Swarm drones for cooperative search

**🚁 Operational Workflow**
Deployment: UAVs launched near avalanche site

Surveillance: Systematic grid scanning with vision and/or IR sensors

Processing: Onboard real-time victim detection

Alert: GPS-tagged victim coordinates sent to SAR teams

Monitoring: Continued scanning & video streaming

**📄 Research Paper**
A detailed paper on this work can be found in the repository under avalanchefinal.doc.

📬 Contact
Author: SK. Yaseen Basha

Email: yaseensk2005@gmail.com

Affiliation: Lovely Professional University, Punjab, India
