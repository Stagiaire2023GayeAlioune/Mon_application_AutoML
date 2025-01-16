import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Charger les données
education_level = pd.read_csv("data/EducationLevel.csv")
employee_data = pd.read_csv("data/Employee.csv")
performance_rating = pd.read_csv("data/PerformanceRating.csv")
rating_level = pd.read_csv("data/RatingLevel.csv")
satisfied_level = pd.read_csv("data/SatisfiedLevel.csv")

# Préparation des données
employee_data['Year'] = pd.to_datetime(employee_data['HireDate']).dt.year
employee_data['AgeGroup'] = pd.cut(employee_data['Age'], 
                                   bins=[0, 19, 29, 39, 49, float('inf')], 
                                   labels=["<20", "20-29", "30-39", "40-49", "50+"])
employee_data['FullName'] = employee_data['FirstName'] + " " + employee_data['LastName']

# Calcul des métriques
total_employees = len(employee_data)
total_active_employees = len(employee_data[employee_data['Attrition'] == "No"])
total_inactive_employees = len(employee_data[employee_data['Attrition'] == "Yes"])
attrition_rate = round((total_inactive_employees / total_employees) * 100, 2)

youngest_age = employee_data['Age'].min()
oldest_age = employee_data['Age'].max()

# Layout principal
st.title("HR Analytics Dashboard")

tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Demographics", "Performance Tracker", "Attrition"])

# **Tab 1: Overview**
with tab1:
    st.header("Overview Metrics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Employees", total_employees)
    col2.metric("Active Employees", total_active_employees)
    col3.metric("Inactive Employees", total_inactive_employees)
    col4.metric("Attrition Rate (%)", attrition_rate)

    st.subheader("Employee Hiring Trends")
    hiring_data = employee_data.groupby(['Year', 'Attrition']).size().reset_index(name='Count')
    fig_hiring_trends = px.bar(hiring_data, x='Year', y='Count', color='Attrition', 
                               barmode='stack', title="Employee Hiring Trends")
    st.plotly_chart(fig_hiring_trends)

    st.subheader("Active Employees by Department")
    active_employees = employee_data[employee_data['Attrition'] == "No"]
    department_counts = active_employees.groupby('Department').size().reset_index(name='Count')
    fig_active_dept = px.bar(department_counts, x='Count', y='Department', 
                             orientation='h', title="Active Employees by Department")
    st.plotly_chart(fig_active_dept)

# **Tab 2: Demographics**
with tab2:
    st.header("Demographics")
    col1, col2 = st.columns(2)
    col1.metric("Youngest Age", youngest_age)
    col2.metric("Oldest Age", oldest_age)

    st.subheader("Employees by Age Group")
    age_group_counts = employee_data['AgeGroup'].value_counts().reset_index()
    age_group_counts.columns = ['AgeGroup', 'Count']
    fig_age_groups = px.bar(age_group_counts, x='AgeGroup', y='Count', title="Employees by Age Group")
    st.plotly_chart(fig_age_groups)

    st.subheader("Employees by Marital Status")
    marital_status_counts = employee_data['MaritalStatus'].value_counts().reset_index()
    marital_status_counts.columns = ['MaritalStatus', 'Count']
    fig_marital_status = px.pie(marital_status_counts, names='MaritalStatus', values='Count', 
                                title="Employees by Marital Status")
    st.plotly_chart(fig_marital_status)

# **Tab 3: Performance Tracker**
with tab3:
    st.header("Performance Tracker")
    employee_names = employee_data['FullName'].unique()
    selected_employee = st.selectbox("Select Employee", employee_names)

    # Filtrer les données pour l'employé sélectionné
    selected_data = employee_data[employee_data['FullName'] == selected_employee]
    selected_reviews = performance_rating[performance_rating['EmployeeID'] == selected_data['EmployeeID'].values[0]]

    st.subheader("Key Dates")
    col1, col2, col3 = st.columns(3)
    col1.metric("Start Date", selected_data['HireDate'].values[0])
    last_review_date = selected_reviews['ReviewDate'].max()
    col2.metric("Last Review", last_review_date)
    col3.metric("Next Review", pd.to_datetime(last_review_date) + pd.DateOffset(years=1))

    st.subheader("Satisfaction Ratings")
    fig_job_satisfaction = px.line(selected_reviews, x='ReviewDate', y='JobSatisfaction', 
                                    title="Job Satisfaction Over Time")
    st.plotly_chart(fig_job_satisfaction)

# **Tab 4: Attrition**
with tab4:
    st.header("Attrition Analysis")
    st.metric("Global Attrition Rate", f"{attrition_rate}%")

    st.subheader("Attrition by Department and Job Role")
    department_job_attrition = employee_data.groupby(['Department', 'JobRole']) \
        .agg(TotalEmployees=('EmployeeID', 'count'), 
             AttritionCount=('Attrition', lambda x: (x == "Yes").sum())) \
        .reset_index()
    department_job_attrition['AttritionRate'] = (department_job_attrition['AttritionCount'] / 
                                                 department_job_attrition['TotalEmployees']) * 100
    fig_attrition = px.bar(department_job_attrition, x='AttritionRate', y='Department', color='JobRole', 
                           title="Attrition Rate by Department and Job Role", orientation='h')
    st.plotly_chart(fig_attrition)

    st.subheader("Attrition by Years at Company")
    tenure_attrition = employee_data.groupby('YearsAtCompany') \
        .agg(TotalEmployees=('EmployeeID', 'count'), 
             AttritionCount=('Attrition', lambda x: (x == "Yes").sum())) \
        .reset_index()
    tenure_attrition['AttritionRate'] = (tenure_attrition['AttritionCount'] / 
                                         tenure_attrition['TotalEmployees']) * 100
    fig_tenure = px.line(tenure_attrition, x='YearsAtCompany', y='AttritionRate', 
                         title="Attrition Rate by Tenure")
    st.plotly_chart(fig_tenure)
print(education_level)