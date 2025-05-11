# Heart Disease Health Indicator

A predictive modeling project focused on identifying individuals at risk of heart disease using health survey data. This project uses various classification algorithms to analyze behavioral, lifestyle, and physiological features and build accurate models to assist in health risk assessment.

---

## Project Overview

Heart disease remains a major health challenge globally. This project uses a large, real-world dataset from the **BRFSS 2015** health survey to predict whether an individual has or has had heart disease, based on a variety of risk factors.

---

## Dataset

- **Source**: Behavioral Risk Factor Surveillance System (BRFSS) 2015
- **Size**: 100,638 cleaned survey responses
- **Target Variable**: `HeartDiseaseorAttack` (Binary: 0 = No, 1 = Yes)
- **Class Distribution**: 
  - 76,745 without heart disease  
  - 23,893 with heart disease

### Key Features Include:
- HighBP, HighChol, CholCheck
- BMI, Smoker, Stroke, Diabetes
- Physical activity, Diet (Fruits/Veggies), Alcohol consumption
- Healthcare access, Mental & Physical health
- Demographics: Age, Sex, Income, Education

---

## Models Used

Three machine learning models were trained and compared:
1. **Decision Tree**
2. **Support Vector Machine (SVM)**
3. **Logistic Regression**

### Evaluation Metrics:
- Accuracy
- F1 Score
- Confusion Matrix
- ROC Curve

---

## Results

| Model                | Accuracy | F1 Score | Misclassification Error |
|---------------------|----------|----------|--------------------------|
| Decision Tree        | 78.3%    | 0.15     | 21.7%                    |
| SVM                  | 80.3%    | 0.09     | 19.7%                    |
| Logistic Regression  | Evaluated via ROC curve only                     |

- SVM achieved the **highest accuracy**, while Decision Tree showed **better class separation**.
- ROC curves were plotted for all models to visualize performance.

---

## Project Workflow

- Data Cleaning and Feature Engineering (BMI categorization, NA removal)
- Data Splitting: 40% Training, 30% Validation, 30% Testing
- Model Training and Evaluation
- ROC Curve Analysis and Tree Pruning
- Comparative Study of Classifiers

---

## Technologies Used

- **Language**: R
- **Libraries**: `tree`, `e1071`, `caret`, `pROC`, `dplyr`, `ggplot2`
- **Environment**: RStudio

---

## Future Improvements

- Handle class imbalance (e.g., SMOTE, resampling)
- Incorporate ensemble models like Random Forest or XGBoost
- Use cross-validation to improve model robustness
- Explore interpretability using SHAP or feature importance

---

## Authors

- **Abhijeet Anand** ([@abhan872](https://github.com/abhijeet16))
- **Hussnain Khalid** ([@huskh803](https://github.com))
