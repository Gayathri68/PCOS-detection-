# PCOS Detection using Machine Learning 🩺💻

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/Library-XGBoost-blue.svg)](https://xgboost.readthedocs.io/)
[![Optuna](https://img.shields.io/badge/Optimization-Optuna-red.svg)](https://optuna.org/)

## 📌 Project Overview
Polycystic Ovary Syndrome (PCOS) is a common health condition affecting women, characterized by hormonal imbalances and metabolic issues. Early and accurate detection of PCOS is critical for effective treatment and management. 

This project uses advanced Machine Learning techniques—including **Random Forest**, **XGBoost**, and hyperparameter optimization via **Optuna**—to predict the presence of PCOS (`PCOS (Y/N)`) based on clinical and physical patient data.

## 📊 Dataset Information
The models are trained on clinical data extracted from two primary sources:
1. `PCOS_infertility.csv`
2. `PCOS_data_without_infertility.xlsx`

The datasets are merged based on the `Patient File No.` to create a comprehensive profile for each patient.
> **Note:** The dataset files might be too large for standard GitHub hosting or kept private for medical confidentiality. If you wish to reproduce this project, please download the required dataset files (e.g., from Kaggle) and place them in the working directory before running the notebook.

### Key Features Used For Prediction
- **Demographics & Physicals:** Age, Weight, Height, BMI, Blood Group, Pulse Rate.
- **Hormonal Indicators:** FSH (mIU/mL), LH (mIU/mL), FSH/LH Ratio, AMH (ng/mL), PRL (ng/mL), Vit D3, TSH.
- **Ultrasound Metrics:** Follicle No. (Left & Right), Avg. Follicle Size, Endometrium thickness.
- **Clinical Symptoms:** Weight gain, Hair growth, Skin darkening, Pimples, Fast food consumption, Regular exercise.

## 🧠 Methodology
1. **Data Preprocessing & Merging:** Combining datasets and aligning patient records.
2. **Feature Selection:** Utilizing `SelectKBest` and `chi2` to identify the most significant predictive features.
3. **Model Building:** Training classification models such as **RandomForestClassifier** and **XGBoost**.
4. **Hyperparameter Tuning:** Implementing **Optuna** to systematically search for optimal model parameters to maximize accuracy and reliability.
5. **Evaluation:** Analyzing model performance using standard accuracy metrics and visualizations via `seaborn` and `matplotlib`.

## ⚙️ Requirements
To run the notebook successfully, you will need the following libraries:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost optuna
```

## 🚀 Usage 
1. Clone this repository to your local machine:
   ```bash
   git clone https://github.com/Gayathri68/PCOS-detection-.git
   ```
2. Navigate to the directory and ensure the datasets (`PCOS_infertility.csv` and `PCOS_data_without_infertility.xlsx`) are available in the root folder.
3. Open the Jupyter Notebook:
   ```bash
   jupyter notebook project.ipynb
   ```
4. Run all cells from top to bottom to observe data preprocessing, training, and model evaluation metrics.

## 🤝 Contributing
Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/Gayathri68/PCOS-detection-/issues).

---
*Developed with ❤️ as an analytical tool for early diagnostic assistance in women's health.*
