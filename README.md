# Multi-Disease-Predictive-Screening-Tool
A machine learning-based decision-support tool for ASHA workers to perform non-invasive screening of heart disease, diabetes, and anemia. Using basic patient data, it predicts risk levels, enabling early detection and timely medical intervention in rural areas.
<img width="836" height="435" alt="image" src="https://github.com/user-attachments/assets/dbb5a31c-3903-4129-84c4-648d77bcb6d7" />


### Tools:
-Anaconda Distribution: For managing the Python environment and packages.
-Jupyter Notebook: For initial data exploration, cleaning, and model training.
-Visual Studio Code: As the primary code editor for script development.

### Architecture/Framework:
Streamlit: An open-source Python framework for building and sharing web apps for machine learning and data science.

### Software Language:
-Python 3.9: The core programming language.
-Libraries: Pandas (data manipulation), Scikit-learn (machine learning), Joblib (model saving), XGBoost (modeling), Streamlit (web framework), Matplotlib & Seaborn (visualization).

### Coding: 
All coding was done in Python 3.9 within an isolated Conda virtual environment (project_env) to ensure stability and reproducibility. The project was structured into two main scripts:
-train_models.ipynb: A Jupyter Notebook used for the entire experimental workflow. This included loading each of the three datasets, performing detailed data cleaning and preprocessing (e.g., handling missing values, one-hot encoding categorical features), training and comparing the Random Forest and XGBoost models, and finally, saving the three champion models and their required column structures to disk using the joblib library.
-dashboard.py: A Python script containing the Streamlit application code. This script is responsible for loading the pre-trained models, building the multi-page user interface, handling user input, preparing the input for the models, and displaying the final predictions and recommendations.

### Testing Report:
-Unit Testing: Individual functions, such as the data cleaning steps and the risk score calculation, were tested with sample inputs to ensure they produced the expected outputs.
-Integration Testing: The primary integration test was ensuring the user inputs from the sidebar were correctly processed and passed to the three different models, each of which has a unique feature set.
-User Acceptance Testing (UAT): The dashboard was iteratively tested by creating hypothetical high-risk and low-risk patient profiles to validate that the predictions were logical and the UI feedback (e.g., color-coding, recommendations) was correct and intuitive. This process uncovered and led to the fixing of several logical flaws, such as the initial model's handling of abnormally high hemoglobin.
-Model Validation: The final champion models achieved the following accuracies on their respective unseen test sets:
■ Heart Disease: 87.89%
■ Diabetes: 76.62%
■ Anemia: 99.00%
<img width="852" height="582" alt="image" src="https://github.com/user-attachments/assets/e0372cc9-736c-4440-9c79-2e37d9033451" />

## Dashborad
**Basic Info Dashboard**
<img width="1253" height="480" alt="image" src="https://github.com/user-attachments/assets/71c0c05a-af58-4a05-8782-1e5dcc12bb47" />
**Advanced Info Dashboard**
<img width="1253" height="555" alt="image" src="https://github.com/user-attachments/assets/6a2a82f3-5ddc-4a2f-bca8-e2edb3af74fd" />
**Add Patient Info in Database**
<img width="1253" height="566" alt="image" src="https://github.com/user-attachments/assets/9ecfc5c5-2de9-4f35-baf6-b143dfb013e8" />
**View Patient Info History**
<img width="1253" height="236" alt="image" src="https://github.com/user-attachments/assets/b7ef7709-33ff-4de6-9014-7b078d265796" />
**Health Recommendation**
<img width="1253" height="599" alt="image" src="https://github.com/user-attachments/assets/e41acaf1-d80d-4d86-93b6-6d75b9d78cf4" />
**Dataset Used Info**
<img width="1253" height="236" alt="image" src="https://github.com/user-attachments/assets/0f4b963c-8a4e-43e0-8b79-9917e32aba3b" />
<img width="1253" height="580" alt="image" src="https://github.com/user-attachments/assets/c785682d-6db5-4bbc-99de-e7a1eb99885a" />
**Project Info**
<img width="1253" height="555" alt="image" src="https://github.com/user-attachments/assets/d4f98afe-39a0-443f-996d-9fc6ebc0425f" />




