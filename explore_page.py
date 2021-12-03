import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def reducing_countries(categories, cutoff):
    categorical_map = {}
    for i in range(len(categories)):
        if categories.values[i] >= cutoff:
            categorical_map[categories.index[i]] = categories.index[i]
        else:
            categorical_map[categories.index[i]] = 'Other'
    return categorical_map

def cleanYears(x):
    if x == 'More than 50 years':
        return 50
    if x == 'Less than 1 year':
        return 0.5
    return float(x)

def cleanEdu(x):
    if 'Bachelor’s degree' in x:
        return 'Bachelor’s degree'
    if 'Master’s degree' in x:
        return 'Master’s degree'
    if 'Professional degree' in x or 'Other doctoral degree' in x:
        return 'Professional degree'
    return 'Less than a Bachelors'

@st.cache                                       #this will save info and not rum this function again and again for same database
def load_data():
    df = pd.read_csv("survey_results_public.csv")
    df = df[["Country","EdLevel","YearsCodePro","Employment","ConvertedComp"]]
    df = df.rename({"ConvertedComp": "Salary"}, axis="columns")
    df = df[df["Salary"].notnull()]
    df = df.dropna()
    df = df[df['Employment'] == 'Employed full-time']
    df = df.drop("Employment", axis= 'columns')
    l = df['Country'].value_counts()
    country_map = reducing_countries(l,400)
    df['Country'] = df['Country'].map(country_map)
    df = df[df['Salary']<=250000]
    df = df[df['Salary']>=10000]
    df = df[df['Country'] != 'Other']
    df['YearsCodePro'] = df['YearsCodePro'].apply(cleanYears)
    df['EdLevel'] = df['EdLevel'].apply(cleanEdu)
    return df

df = load_data()


def show_explore_page():
    st.title("Explore Software Engineer Salaries")
    st.write(""" ### Stack Overflow Developer Survey 2020 """)
    data = df["Country"].value_counts()
    fig, ax1 = plt.subplots()
    ax1.pie(data, labels= data.index, autopct="%1.1f%%", shadow=True, startangle= 90)
    ax1.axis("equal")

    st.write("""### Number of Data from different Countries """)
    st.pyplot(fig)

    st.write(""" ### Mean Salary Based on Country """)
    data = df.groupby(["Country"])["Salary"].mean().sort_values(ascending=True)
    st.bar_chart(data)

    st.write(""" ### Mean Salary Based on Experience """)
    data = df.groupby(["YearsCodePro"])["Salary"].mean().sort_values(ascending=True)
    st.line_chart(data)
