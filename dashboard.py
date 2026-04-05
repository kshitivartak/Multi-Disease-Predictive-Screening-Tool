from pathlib import Path
import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================================================================
# PAGE CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="ASHA Health Screening Tool",
    page_icon="👩‍⚕️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# LOAD THE TRAINED MODELS
# ==============================================================================
@st.cache_resource
def load_models():
    try:
        models = {
            "heart": {"model": joblib.load('heart_disease_model.joblib'), "cols": joblib.load('heart_disease_columns.joblib')},
            "diabetes": {"model": joblib.load('diabetes_model.joblib'), "cols": joblib.load('diabetes_columns.joblib')},
            "anemia": {"model": joblib.load('anemia_model.joblib'), "cols": joblib.load('anemia_columns.joblib')}
        }
        return models
    except FileNotFoundError as e:
        st.error(f"ERROR: A model file was not found. Please ensure all .joblib files are in the same folder. Missing file: {e.filename}")
        st.stop()

models = load_models()

# ==============================================================================
# PAGE 1: PREDICTION TOOL (FINAL POLISHED VERSION)
# ==============================================================================
def prediction_page():
    st.title("👩‍⚕️ Basic Patient Risk Prediction")
    st.markdown("Enter a patient's key vitals in the sidebar for a quick risk assessment.")

    # --- Sidebar for Patient Input with NEW, CLEANER LAYOUT AND TOOLTIPS ---
    with st.sidebar:
        st.header("Enter Patient Vitals (Basic)")
        
        with st.expander("👤 General Information", expanded=True):
            age = st.slider('Age', 18, 100, 50, key="simple_age", help="Patient's age in years.")
            sex_option = st.selectbox('Sex', ('Female', 'Male'), key="simple_sex", help="Patient's biological sex.")
            BMI = st.slider('Body Mass Index (BMI)', 15.0, 70.0, 25.0, help="Calculated as kg/m^2. Normal range: 18.5-24.9.", key="simple_bmi")
            if sex_option == 'Female':
                Pregnancies = st.slider('Number of Pregnancies', 0, 20, 1, key="simple_preg", help="Total number of times pregnant.")
            else:
                Pregnancies = 0
                st.write("Pregnancies: 0 (N/A for Male)")

        with st.expander("❤️ Key Vitals", expanded=True):
            trestbps = st.slider('Blood Pressure (Systolic, mm Hg)', 80, 200, 120, key="simple_bp", help="The top number in a blood pressure reading. Normal is < 120.")
            Glucose = st.slider('Glucose Level (mg/dL)', 50, 250, 100, key="simple_glucose", help="Blood sugar level. Normal is < 140 mg/dL.")
            Hemoglobin = st.slider('Hemoglobin (g/dL)', 6.0, 18.0, 13.0, key="simple_hb", help="Normal (Women): 12.1-15.1, (Men): 13.8-17.2")
            chest_pain_option = st.selectbox('Do they experience chest pain?', ('No', 'Yes'), key="simple_cp", help="Does the patient report experiencing chest pain?")
        
        # (The patient_profile dictionary creation remains the same)
        patient_profile = {'age': age, 'sex': sex_option, 'BMI': BMI, 'Pregnancies': Pregnancies, 'trestbps': trestbps, 'Glucose': Glucose, 'Hemoglobin': Hemoglobin,'cp': 'asymptomatic' if chest_pain_option == 'No' else 'typical angina','chol': 200, 'thalach': 150, 'fbs': True if Glucose > 120 else False, 'exang': True if chest_pain_option == 'Yes' else False,'restecg': 'normal', 'oldpeak': 1.0, 'slope': 'flat', 'ca': 0.0, 'thal': 'normal', 'BloodPressure': trestbps,'SkinThickness': 29, 'Insulin': 125, 'DiabetesPedigreeFunction': 0.47, 'Age': age, 'Gender': 1 if sex_option == 'Male' else 0,'MCH': 28, 'MCHC': 33, 'MCV': 90}

    # (The rest of the prediction_page function remains exactly the same)
    st.header("Patient Assessment")
    button_cols = st.columns([1, 1, 5]); get_assessment = button_cols[0].button('Get Health Assessment', key='simple_get'); clear_results = button_cols[1].button('Clear Results', key='simple_clear')
    if 'show_results_simple' not in st.session_state: st.session_state.show_results_simple = False
    if get_assessment: st.session_state.show_results_simple = True
    if clear_results: st.session_state.show_results_simple = False

    if st.session_state.show_results_simple:
        model_risk_flags = []; vitals_risk_flags = []
        with st.expander("Show Patient Input Summary", expanded=True):
            summary_cols = st.columns(4); summary_cols[0].metric("Age", patient_profile['age']); summary_cols[1].metric("Sex", patient_profile['sex']); summary_cols[2].metric("BMI", f"{patient_profile['BMI']:.1f}"); summary_cols[3].metric("Blood Pressure", f"{patient_profile['trestbps']}")
            indicator_cols = st.columns(3)
            with indicator_cols[0]:
                glucose_val = patient_profile['Glucose'];
                if glucose_val > 140: st.error(f"**Glucose:** {glucose_val} mg/dL (High)"); vitals_risk_flags.append("High Glucose")
                elif glucose_val > 100: st.warning(f"**Glucose:** {glucose_val} mg/dL (Elevated)")
                else: st.success(f"**Glucose:** {glucose_val} mg/dL (Normal)")
            with indicator_cols[1]:
                chol_val = patient_profile['chol'];
                if chol_val > 239: st.error(f"**Cholesterol:** {chol_val} mg/dL (High)"); vitals_risk_flags.append("High Cholesterol")
                elif chol_val > 200: st.warning(f"**Cholesterol:** {chol_val} mg/dL (Borderline High)")
                else: st.success(f"**Cholesterol:** {chol_val} mg/dL (Normal)")
            with indicator_cols[2]:
                hb_val = patient_profile['Hemoglobin']; gender = patient_profile['Gender']; hemoglobin_status = "Normal"
                if gender == 0 and (hb_val < 12.1 or hb_val > 15.1): hemoglobin_status = "Low" if hb_val < 12.1 else "High"
                elif gender == 1 and (hb_val < 13.8 or hb_val > 17.2): hemoglobin_status = "Low" if hb_val < 13.8 else "High"
                if hemoglobin_status != "Normal": st.warning(f"**Hemoglobin:** {hb_val:.2f} g/dL ({hemoglobin_status})"); vitals_risk_flags.append(f"{hemoglobin_status} Hemoglobin")
                else: st.success(f"**Hemoglobin:** {hb_val:.2f} g/dL (Normal)")
        
        st.markdown("---"); st.subheader("Model Prediction Results"); res_col1, res_col2, res_col3 = st.columns(3)
        
        with res_col1:
            st.markdown("##### ❤️ Heart Disease")
            heart_data_raw = pd.DataFrame([patient_profile]); heart_data_raw['sex'] = heart_data_raw['sex'].apply(lambda x: 1 if x == 'Male' else 0); heart_data_processed = pd.get_dummies(heart_data_raw); heart_data_final = heart_data_processed.reindex(columns=models['heart']['cols'], fill_value=0)
            heart_risk_proba = models['heart']['model'].predict_proba(heart_data_final)[0][1]
            heart_score = heart_risk_proba * 100
            if heart_score > 70: st.error(f"**Status:** HIGH RISK"); model_risk_flags.append("Heart Disease")
            elif heart_score > 40: st.warning(f"**Status:** MODERATE RISK")
            else: st.success(f"**Status:** LOW RISK")
            st.write(f"**Risk Score:** {heart_score:.1f}%")
        
        with res_col2:
            st.markdown("##### 🩸 Diabetes")
            diabetes_data = pd.DataFrame([patient_profile], columns=models['diabetes']['cols'])
            diabetes_risk_proba = models['diabetes']['model'].predict_proba(diabetes_data)[0][1]
            diabetes_score = diabetes_risk_proba * 100
            if diabetes_score > 70: st.error(f"**Status:** HIGH RISK"); model_risk_flags.append("Diabetes")
            elif diabetes_score > 40: st.warning(f"**Status:** MODERATE RISK")
            else: st.success(f"**Status:** LOW RISK")
            st.write(f"**Risk Score:** {diabetes_score:.1f}%")

        with res_col3:
            st.markdown("##### 🔬 Anemia")
            anemia_data = pd.DataFrame([patient_profile], columns=models['anemia']['cols'])
            anemia_risk_proba = models['anemia']['model'].predict_proba(anemia_data)[0][1]
            anemia_score = anemia_risk_proba * 100
            if anemia_score > 70: st.error(f"**Status:** HIGH RISK"); model_risk_flags.append("Anemia (model)")
            elif anemia_score > 40: st.warning(f"**Status:** MODERATE RISK")
            else: st.success(f"**Status:** LOW RISK")
            st.write(f"**Risk Score:** {anemia_score:.1f}%")

        st.markdown("---"); st.subheader("Overall Assessment & Recommendation"); final_risk_flags = sorted(list(set(model_risk_flags + vitals_risk_flags)))
        if not final_risk_flags: st.success("✅ **Overall Status: Low Risk.** Recommend routine follow-up.")
        else: st.error(f"⚠️ **Overall Status: PRIORITY ALERT.** High risk detected. **Reason(s):** {', '.join(final_risk_flags)}. Immediate follow-up is strongly recommended.")
        
        st.markdown("---")
        if st.button("Generate Patient Report"):
            st.session_state.open_report_modal = True
        if 'open_report_modal' not in st.session_state: st.session_state.open_report_modal = False
        if st.session_state.open_report_modal:
            with st.form("report_form"):
                st.subheader("Enter Additional Patient Details for Report")
                patient_name = st.text_input("Patient Name"); patient_location = st.text_input("Village / Town"); smoker_status = st.selectbox("Smoker Status", ("No", "Yes", "Former")); alcohol_consumption = st.selectbox("Alcohol Consumption", ("No", "Occasionally", "Regularly")); diet_quality = st.selectbox("Diet Quality", ("Healthy (Mostly home-cooked)", "Moderate", "Mostly Fast Food")); activity_level = st.selectbox("Physical Activity", ("Active (Daily exercise)", "Moderate (Walks etc.)", "Sedentary (Mostly sitting)"))
                submitted = st.form_submit_button("Save Report")
                if submitted:
                    report = {'Date': pd.to_datetime('today').strftime('%Y-%m-%d'), 'Patient Name': patient_name, 'Location': patient_location, 'Age': patient_profile['age'], 'Sex': patient_profile['sex'], 'BMI': f"{patient_profile['BMI']:.1f}", 'Blood Pressure': patient_profile['trestbps'], 'Glucose': patient_profile['Glucose'], 'Hemoglobin': f"{patient_profile['Hemoglobin']:.2f}", 'Heart Disease Risk (%)': f"{heart_score:.1f}", 'Diabetes Risk (%)': f"{diabetes_score:.1f}", 'Anemia Risk (%)': f"{anemia_score:.1f}", 'Overall Recommendation': "Priority Alert" if final_risk_flags else "Low Risk", 'Reasons for Alert': ", ".join(final_risk_flags) if final_risk_flags else "N/A", 'Smoker Status': smoker_status, 'Alcohol Consumption': alcohol_consumption, 'Diet Quality': diet_quality, 'Activity Level': activity_level}
                    history_file = Path('patient_history.csv')
                    new_report_df = pd.DataFrame([report])
                    if not history_file.is_file(): new_report_df.to_csv(history_file, index=False)
                    else: new_report_df.to_csv(history_file, mode='a', header=False, index=False)
                    st.success(f"Report for {patient_name} saved successfully!"); st.session_state.open_report_modal = False; st.rerun()
                    
# ==============================================================================
# PAGE 2: ADVANCED PREDICTION TOOL
# ==============================================================================
def advanced_prediction_page():
    st.title("⚙️ Advanced Patient Risk Prediction")
    st.markdown("Use this tool if you have more detailed clinical data for a more nuanced prediction.")

    with st.sidebar:
        st.header("Enter Patient Vitals (Advanced)")
        with st.expander("👤 General Information", expanded=True):
            age_adv = st.slider('Age', 18, 100, 50, key="adv_age", help="Patient's age in years.")
            sex_option_adv = st.selectbox('Sex', ('Female', 'Male'), key="adv_sex", help="Patient's biological sex.")
            BMI_adv = st.slider('Body Mass Index (BMI)', 15.0, 70.0, 25.0, key="adv_bmi", help="Calculated as kg/m^2. Normal range: 18.5-24.9.")
            if sex_option_adv == 'Female':
                Pregnancies_adv = st.slider('Number of Pregnancies', 0, 20, 1, key="adv_preg", help="Total number of times pregnant.")
            else:
                Pregnancies_adv = 0
                st.write("Pregnancies: 0 (N/A for Male)")
        
        with st.expander("❤️ Cardiovascular Panel", expanded=True):
            cp_adv = st.selectbox('Chest Pain Type', ('typical angina', 'atypical angina', 'non-anginal', 'asymptomatic'), key="adv_cp", help="The type of chest pain experienced by the patient.")
            trestbps_adv = st.slider('Resting Blood Pressure', 80, 200, 120, key="adv_bp", help="The top number in a blood pressure reading. Normal is < 120.")
            chol_adv = st.slider('Cholesterol', 100, 600, 200, key="adv_chol", help="Serum cholesterol in mg/dl. Desirable is < 200.")
            thalach_adv = st.slider('Max Heart Rate Achieved', 60, 220, 150, key="adv_thalach", help="The highest heart rate achieved during a stress test.")
            exang_adv = st.selectbox('Exercise Induced Angina?', ('No', 'Yes'), key="adv_exang", help="Does the patient experience chest pain during exercise?")
            restecg_adv = st.selectbox('Resting ECG', ('normal', 'st-t abnormality', 'lv hypertrophy'), key="adv_restecg", help="Resting electrocardiographic results.")
            oldpeak_adv = st.slider('ST Depression (oldpeak)', 0.0, 6.5, 1.0, key="adv_oldpeak", help="ST depression induced by exercise relative to rest.")
            thal_adv = st.selectbox('Thalassemia Stress Test', ('normal', 'fixed defect', 'reversable defect'), key="adv_thal", help="Result of Thalassemia stress test. 'Reversable defect' is a high-risk indicator.")

        with st.expander("🩸 Metabolic & Blood Panel", expanded=True):
            Glucose_adv = st.slider('Glucose Level (mg/dL)', 50, 250, 100, key="adv_glucose", help="Blood sugar level. Normal is < 140 mg/dL.")
            Hemoglobin_adv = st.slider('Hemoglobin (g/dL)', 6.0, 18.0, 13.0, key="adv_hb", help="Normal (Women): 12.1-15.1, (Men): 13.8-17.2")
            MCH_adv = st.slider('MCH', 15.0, 35.0, 28.0, key="adv_mch", help="Mean Corpuscular Hemoglobin.")
            MCHC_adv = st.slider('MCHC', 25.0, 40.0, 32.0, key="adv_mchc", help="Mean Corpuscular Hemoglobin Concentration.")
            MCV_adv = st.slider('MCV', 70.0, 110.0, 90.0, key="adv_mcv", help="Mean Corpuscular Volume.")
        
        patient_profile_adv = {'age': age_adv, 'sex': sex_option_adv, 'BMI': BMI_adv, 'Pregnancies': Pregnancies_adv, 'trestbps': trestbps_adv, 'chol': chol_adv, 'Glucose': Glucose_adv, 'Hemoglobin': Hemoglobin_adv, 'cp': cp_adv, 'thalach': thalach_adv, 'exang': True if exang_adv == 'Yes' else False, 'restecg': restecg_adv, 'fbs': True if Glucose_adv > 120 else False, 'oldpeak': oldpeak_adv, 'thal': thal_adv, 'slope': 'flat', 'ca': 0.0, 'BloodPressure': trestbps_adv, 'SkinThickness': 29, 'Insulin': 125, 'DiabetesPedigreeFunction': 0.47, 'Age': age_adv, 'Gender': 1 if sex_option_adv == 'Male' else 0, 'MCH': MCH_adv, 'MCHC': MCHC_adv, 'MCV': MCV_adv}

    st.header("Advanced Patient Assessment")
    if st.button("Get Advanced Assessment"):
        # (The entire results display section for the advanced page)
        model_risk_flags_adv = []; vitals_risk_flags_adv = []
        with st.expander("Show Patient Input Summary", expanded=True):
            summary_cols = st.columns(4); summary_cols[0].metric("Age", patient_profile_adv['age']); summary_cols[1].metric("Sex", patient_profile_adv['sex']); summary_cols[2].metric("BMI", f"{patient_profile_adv['BMI']:.1f}"); summary_cols[3].metric("Blood Pressure", f"{patient_profile_adv['trestbps']}")
            indicator_cols = st.columns(3)
            with indicator_cols[0]:
                glucose_val = patient_profile_adv['Glucose'];
                if glucose_val > 140: st.error(f"**Glucose:** {glucose_val} mg/dL (High)"); vitals_risk_flags_adv.append("High Glucose")
                elif glucose_val > 100: st.warning(f"**Glucose:** {glucose_val} mg/dL (Elevated)")
                else: st.success(f"**Glucose:** {glucose_val} mg/dL (Normal)")
            with indicator_cols[1]:
                chol_val = patient_profile_adv['chol']
                if chol_val > 239: st.error(f"**Cholesterol:** {chol_val} mg/dL (High)"); vitals_risk_flags_adv.append("High Cholesterol")
                elif chol_val > 200: st.warning(f"**Cholesterol:** {chol_val} mg/dL (Borderline High)")
                else: st.success(f"**Cholesterol:** {chol_val} mg/dL (Normal)")
            with indicator_cols[2]:
                hb_val = patient_profile_adv['Hemoglobin']; gender = patient_profile_adv['Gender']; hemoglobin_status = "Normal"
                if gender == 0 and (hb_val < 12.1 or hb_val > 15.1): hemoglobin_status = "Low" if hb_val < 12.1 else "High"
                elif gender == 1 and (hb_val < 13.8 or hb_val > 17.2): hemoglobin_status = "Low" if hb_val < 13.8 else "High"
                if hemoglobin_status != "Normal": st.warning(f"**Hemoglobin:** {hb_val:.2f} g/dL ({hemoglobin_status})"); vitals_risk_flags_adv.append(f"{hemoglobin_status} Hemoglobin")
                else: st.success(f"**Hemoglobin:** {hb_val:.2f} g/dL (Normal)")
        st.markdown("---"); st.subheader("Model Prediction Results"); res_col1, res_col2, res_col3 = st.columns(3)
        def display_prediction_adv(proba, col, disease_name):
            risk_score = proba * 100
            if risk_score > 70: col.error(f"**Status:** HIGH RISK"); model_risk_flags_adv.append(disease_name)
            elif risk_score > 40: col.warning(f"**Status:** MODERATE RISK")
            else: col.success(f"**Status:** LOW RISK")
            col.write(f"**Risk Score:** {risk_score:.1f}%")
        with res_col1:
            st.markdown("##### ❤️ Heart Disease")
            heart_data_raw = pd.DataFrame([patient_profile_adv]); heart_data_raw['sex'] = heart_data_raw['sex'].apply(lambda x: 1 if x == 'Male' else 0); heart_data_processed = pd.get_dummies(heart_data_raw); heart_data_final = heart_data_processed.reindex(columns=models['heart']['cols'], fill_value=0)
            heart_risk_proba = models['heart']['model'].predict_proba(heart_data_final)[0][1]
            display_prediction_adv(heart_risk_proba, st, "Heart Disease")
        with res_col2: st.markdown("##### 🩸 Diabetes"); diabetes_data = pd.DataFrame([patient_profile_adv], columns=models['diabetes']['cols']); diabetes_risk_proba = models['diabetes']['model'].predict_proba(diabetes_data)[0][1]; display_prediction_adv(diabetes_risk_proba, st, "Diabetes")
        with res_col3: st.markdown("##### 🔬 Anemia"); anemia_data = pd.DataFrame([patient_profile_adv], columns=models['anemia']['cols']); anemia_risk_proba = models['anemia']['model'].predict_proba(anemia_data)[0][1]; display_prediction_adv(anemia_risk_proba, st, "Anemia (model)")
        st.markdown("---"); st.subheader("Overall Assessment & Recommendation"); final_risk_flags = sorted(list(set(model_risk_flags_adv + vitals_risk_flags_adv)))
        if not final_risk_flags: st.success("✅ **Overall Status: Low Risk.** The patient's vitals are within normal ranges and models do not predict high risk. Recommend routine follow-up.")
        else: st.error(f"⚠️ **Overall Status: PRIORITY ALERT.** High risk detected. **Reason(s):** {', '.join(final_risk_flags)}. This patient should be prioritized for immediate follow-up and consultation with a qualified doctor is strongly recommended.")

# ==============================================================================
# PAGE 3, 4, 5 (Recommendations, Explorer, About)
# ==============================================================================
def recommendations_page():
    st.title("💡 Health Recommendations")
    st.markdown("This page provides general lifestyle and dietary advice for individuals at risk. This is not a substitute for professional medical advice.")
    st.header("❤️ Heart Disease"); st.expander("See Recommendations for Heart Disease", expanded=True).markdown("- **Diet:** Reduce intake of sodium (salt), saturated fats (red meat, full-fat dairy), and trans fats (processed foods). Increase intake of fruits, vegetables, whole grains, and omega-3s (fish).\n- **Lifestyle:** Aim for 150+ minutes of moderate aerobic exercise per week. Quit smoking. Limit alcohol.\n- **Monitoring:** Regularly check your blood pressure and cholesterol levels.")
    st.header("🩸 Diabetes"); st.expander("See Recommendations for Diabetes", expanded=True).markdown("- **Diet:** Control carbohydrate intake, focusing on complex carbs (whole grains) and fiber. Practice portion control.\n- **Lifestyle:** Regular physical activity is crucial for managing blood sugar. Maintain a healthy weight.\n- **Monitoring:** Regularly monitor blood glucose as advised by your doctor.")
    st.header("🔬 Anemia"); st.expander("See Recommendations for Anemia", expanded=True).markdown("- **Diet:** Increase intake of iron-rich foods like red meat, poultry, fish, beans, lentils, tofu, spinach, and fortified cereals.\n- **Enhance Absorption:** Consume Vitamin C (oranges, strawberries) with iron-rich meals.\n- **Avoid Inhibitors:** Drink tea and coffee between meals, not with them.")

@st.cache_data
def load_dataframe(name):
    if name == "Heart Disease": return pd.read_csv(r"C:\Users\Kshiti\Downloads\Field Project Sem 5\heart_disease.csv")
    elif name == "Diabetes": return pd.read_csv(r"C:\Users\Kshiti\Downloads\Field Project Sem 5\diabetes.csv")
    else: return pd.read_csv(r"C:\Users\Kshiti\Downloads\Field Project Sem 5\anemia.csv")

def exploration_page():
    st.title("📊 Dataset Explorer")
    st.markdown("Select a dataset to explore its features and distributions.")
    dataset_name = st.selectbox("Select a dataset:", ["Heart Disease", "Diabetes", "Anemia"])
    df_explore = load_dataframe(dataset_name)
    st.markdown(f"### Exploring the {dataset_name} Dataset"); st.dataframe(df_explore.head()); st.markdown("---"); st.subheader("Feature Distribution")
    feature = st.selectbox("Select a feature to visualize:", df_explore.columns)
    fig, ax = plt.subplots(); sns.histplot(df_explore[feature], kde=True, ax=ax, bins=30); ax.set_title(f"Distribution of '{feature}'"); st.pyplot(fig)

def about_page():
    st.title("📖 About This Project")
    st.markdown("""This project was designed to create a functional proof-of-concept for a multi-disease screening tool that could be used by community health workers (like ASHA workers) in rural settings. ### Project Goal The primary goal is to leverage machine learning to provide an early warning system for three prevalent health conditions, allowing for the prioritization of at-risk individuals. ### Models and Datasets The application uses three separate machine learning models, each trained on a specialized, public dataset from Kaggle to ensure high accuracy for its specific task.""")
    st.metric("❤️ Heart Disease Model Accuracy", "87.89%"); st.metric("🩸 Diabetes Model Accuracy", "76.62%"); st.metric("🔬 Anemia Model Accuracy", "99.00%")
    st.info("This application is for educational purposes and is not a substitute for professional medical advice.")


# ==============================================================================
# NEW 6: PATIENT HISTORY
# ==============================================================================
def history_page():
    st.title("📂 Patient Assessment History")
    st.markdown("This page shows all previously saved patient assessments.")
    history_file = Path('patient_history.csv')
    if history_file.is_file():
        history_df = pd.read_csv(history_file)
        if history_df.empty: st.info("No patient records have been saved yet.")
        else: st.dataframe(history_df)
    else:
        st.info("No patient records have been saved yet.")

# ==============================================================================
# NAVIGATION
# ==============================================================================
st.sidebar.title("Navigation")
page_options = ["Prediction Tool", "Advanced Prediction Tool", "Health Recommendations","Patient History", "Dataset Explorer", "About Project"]
page = st.sidebar.radio("Go to", page_options)

if page == "Prediction Tool": prediction_page()
elif page == "Advanced Prediction Tool": advanced_prediction_page()
elif page == "Health Recommendations": recommendations_page()
elif page == "Patient History": history_page()
elif page == "Dataset Explorer": exploration_page()
else: about_page()

