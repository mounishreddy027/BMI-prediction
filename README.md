# BMI and Gender Prediction Using Facial Images 🧠📸

This mini-project uses machine learning and computer vision to predict **Body Mass Index (BMI)** categories and analyze inmate offenses using image data and structured datasets.

---

## 📌 Objectives

- Predict BMI from **facial images** (front & side views).
- Categorize individuals into **Underweight, Normal, or Overweight**.
- Analyze **offense types** from inmate data.
- (Optional/Future Scope) Predict gender based on facial features.

---

## 🗂️ Dataset

- CSV file with: `id`, `weight`, `height`
- Frontal and side view facial images
- Offense dataset: `id`, `offense`
- ~10,000 samples used

---

## 🔧 Preprocessing

- Convert weight (lbs → kg) and height (inches → meters)
- Calculate BMI using:
- Categorize BMI:
- Underweight: < 18.5
- Normal: 18.5 ≤ BMI < 24.9
- Overweight: ≥ 25
- Resize all images to **128x128**
- Handle missing images with zero arrays

---

## 🧠 Model Architecture

- Dual-input **Convolutional Neural Network (CNN)**
- One branch for **front** image
- One branch for **side** image
- Flatten + Dense layers → Final classification
- Categorical output: [Underweight, Normal, Overweight]

---

## 🏋️ Training & Evaluation

- Train-test split: **80% training**, **20% testing**
- Optimizer: Adam
- Loss: Categorical Crossentropy
- Metrics:
- ✅ Accuracy
- 📉 Mean Absolute Error (MAE)
- 📉 Mean Squared Error (MSE)
- 📈 R² Score
- 🔗 Pearson Correlation

---

## 📊 Results

- Training Accuracy: ~94.5%
- Testing Accuracy: ~92.3%
- MAE: 0.08
- MSE: 0.12
- R²: 0.89
- Pearson Coefficient: 0.92

---

## 📈 Offense Analysis

- Plotted top 20 inmate offenses
- Visual insights using bar charts (`matplotlib`)

---

## 📂 File Structure


---

## 🚀 How to Run

1. Clone the repo:
   ```bash
   git clone https://github.com/yourusername/bmi-prediction-project.git
   cd bmi-prediction-project
