# 🧠 Demetia Detection Using EEG Signals

A deep learning-based pipeline for automated classification of Alzheimer's Disease (AD), Frontotemporal Dementia (FTD), and Cognitively Normal (CN) individuals using EEG data.

This project transforms raw EEG signals into **Smoothed Pseudo-Wigner–Ville Distribution (SPWVD)** spectrograms and classifies them using an **AlexNet-inspired Convolutional Neural Network**. Leveraging bipolar montage optimization and ensemble learning, the model achieves up to **79% accuracy** on validation data.

---

## 🧰 Tech Stack

- **Programming**: Python, MATLAB
- **Frameworks**: TensorFlow, Keras, scikit-learn
- **Visualization**: Matplotlib, Seaborn
- **Model**: AlexNet-based CNN
- **Preprocessing**: SPWVD spectrograms, bipolar EEG montages
- **Evaluation**: Stratified K-Fold Cross-Validation (10-fold)

---

## 📂 Dataset

📥 https://openneuro.org/datasets/ds004504/versions/1.0.7
