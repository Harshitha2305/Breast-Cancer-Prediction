# 🧠 Breast Cancer Image Classification using CNN (CancerNet)

This project implements a deep learning approach using **Convolutional Neural Networks (CNNs)** to classify **histopathology images** for **breast cancer detection**. The model predicts whether an image contains signs of **Invasive Ductal Carcinoma (IDC)** — the most common form of breast cancer — based on the **IDC dataset** from Kaggle.

---

## 📌 Project Objective

To develop a reliable image classification model that can assist in early breast cancer diagnosis, potentially supporting medical professionals in detecting cancerous tissues from microscopic images.

---

## 📚 Dataset Overview

- **Name:** Breast Histopathology Images (IDC dataset)
- **Source:** [Kaggle – Breast Cancer Histopathological Dataset](https://www.kaggle.com/datasets/paultimothymooney/breast-histopathology-images)
- **Images:** 50x50 pixel RGB images of breast tissue
- **Labels:**
  - `0`: No IDC (Benign)
  - `1`: IDC Present (Malignant)
- **Total Samples (filtered):** ~50,000+ images (subset used for faster processing)

---

## 🧪 Methodology

### 🔍 Preprocessing
- Directory traversal to load image paths and labels
- Resizing and normalizing images
- Converting data into NumPy arrays
- Saving data as `.npy` for faster reloading

### 🧠 CNN Architecture (CancerNet)
- Input layer (50x50x3)
- Convolutional layers with ReLU and MaxPooling
- Dropout layers to prevent overfitting
- Flatten + Dense output layer with sigmoid activation

### 🔀 Train/Test Split
- Stratified split: `80% training` and `20% testing`
- Balance maintained between IDC and non-IDC images

### 🏋️ Model Training
- Binary classification using `binary_crossentropy`
- Optimizer: Adam
- Batch size: 32
- Epochs: 5–10
- Metrics tracked: Accuracy, Loss

---

## 📈 Evaluation Metrics

- **Accuracy**
- **Loss Curve**
- **Confusion Matrix**
- **Classification Report** (Precision, Recall, F1 Score)

---

## 🎯 Key Results

| Epochs | Training Accuracy | Validation Accuracy |
|--------|-------------------|---------------------|
| 5      | ~86%              | ~85%                |
| 10     | ~91%              | ~89–92%             |

> 📊 Confusion matrix and classification report included in notebook for better analysis.

---

## 📌 Performance Insights

- **Model Type:** CNN performed efficiently for this task.
- **Alternative Models:** Pre-trained CNNs (e.g., ResNet50, EfficientNet) may offer better results but require more compute.
- **Model Fit:** No overfitting observed; training and validation curves follow similar trends.
- **Limitations:** Class imbalance, small image size, noise in labels.

---

## 💭 Real-Life Application Scenarios

1. **Hospital Diagnostic Software:** Assist radiologists in preliminary diagnosis from biopsy images.
2. **Remote Pathology Tools:** Telemedicine platforms that support image upload and automatic classification.
3. **Medical Research:** Analyzing large datasets for trends in cancerous cell patterns.
4. **Mobile Health Apps:** Screen tissue images using portable microscopes and mobile apps.

---

## 🧪 How to Run the Project

### 🔧 Setup

```bash
git clone https://github.com/yourusername/breast-cancer-cnn.git
cd breast-cancer-cnn
pip install -r requirements.txt

🗂️ Project Structure
📁 Breast_Cancer/
│
├── Breast_Cancer.ipynb          # Main Jupyter notebook
├── X_data.npy / y_data.npy      # Preprocessed data arrays
├── IDC_dataset/                 # Raw dataset images (download separately)
├── requirements.txt             # Python dependencies
├── MIT License.txt              # License
├── README.md                    # Project documentation
├── .gitignore                   # Git ignore list

🔐 License
This project is licensed under the MIT License. You are free to use, modify, and distribute it with attribution.

See MIT License.txt for details.

🙌 Acknowledgments
Kaggle Dataset

TensorFlow/Keras for the deep learning framework

Python and open-source contributors

🤝 Contributions
Pull requests are welcome. Feel free to fork this repo, make changes, and submit improvements.

