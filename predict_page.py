import streamlit as st
import pickle
import numpy as np
def load_model():
    with open('final_salary_prediction_model.pickle', 'rb') as f:
        data = pickle.load(f)
    return data

data = load_model()

regressor = data['model']
country = data['le_country']
education = data['le_education']

def show_predict_page():
    st.title('Software Developer Salary Prediction')
    st.write("""### We need some information to predict the salary""")
    countries = (
        'United States',
        'United Kingdom',
        'Spain',
        'Netherlands',
        'Germany',
        'Canada'
        'Italy',
        'Brazil',
        'France',
        'India',
        'Sweden',
        'Poland',
        'Australia',
        'Russian Federation'
    )

    educations = (
        'Bachelor’s degree',
        'Master’s degree',
        'Less than a Bachelors',
        'Professional degree'
    )

    country1 = st.selectbox("Country", countries)
    education1 = st.selectbox("Education", educations)
    experi = st.slider("Years of Experience", 0, 50 ,3)

    ok = st.button("Calculate Salary")
    if ok:
        X = np.array([[country1,education1,experi]])
        X[:,0] = country.transform(X[:,0])
        X[:,1] = education.transform(X[:,1])
        X = X.astype(float)

        salary = regressor.predict(X)
        st.subheader(f"The estimated Salary is ${salary[0]:.2f}")

