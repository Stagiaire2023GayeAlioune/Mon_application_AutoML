import streamlit as st
import pandas as pd
import plotly.express as px

# Charger les données
niveau_education = pd.read_csv("data/EducationLevel.csv")
donnees_employes = pd.read_csv("data/Employee.csv")
evaluation_performance = pd.read_csv("data/PerformanceRating.csv")
niveaux_evaluation = pd.read_csv("data/RatingLevel.csv")
niveaux_satisfaction = pd.read_csv("data/SatisfiedLevel.csv")

# Préparation des données
donnees_employes['Annee'] = pd.to_datetime(donnees_employes['HireDate']).dt.year
donnees_employes['GroupeAge'] = pd.cut(
    donnees_employes['Age'], 
    bins=[0, 19, 29, 39, 49, float('inf')], 
    labels=["<20", "20-29", "30-39", "40-49", "50+"]
)
donnees_employes['NomComplet'] = donnees_employes['FirstName'] + " " + donnees_employes['LastName']

# Calcul des métriques
total_employes = len(donnees_employes)
employes_actifs = len(donnees_employes[donnees_employes['Attrition'] == "No"])
employes_inactifs = len(donnees_employes[donnees_employes['Attrition'] == "Yes"])
taux_attrition = round((employes_inactifs / total_employes) * 100, 2)

plus_jeune = donnees_employes['Age'].min()
plus_age = donnees_employes['Age'].max()

# Interface principale
st.title("Tableau de bord d’analyse RH")
st.header("Exploration des données sur la main-d’œuvre, y compris les tendances d’embauche, les données démographiques, les performances et les mesures d’attrition. Conçu pour que les équipes RH puissent surveiller et prendre des décisions basées sur les données.")
         

onglet1, onglet2, onglet3, onglet4 = st.tabs(["Vue d'ensemble", "Démographie", "Suivi de performance", "Attrition"])

# **Onglet 1 : Vue d'ensemble**
with onglet1:
    st.header("Métriques globales")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total des employés", total_employes)
    col2.metric("Employés actifs", employes_actifs)
    col3.metric("Employés inactifs", employes_inactifs)
    col4.metric("Taux d'attrition (%)", taux_attrition)

    st.subheader("Tendances d'embauche des employés")
    embauche_donnees = donnees_employes.groupby(['Annee', 'Attrition']).size().reset_index(name='Total')
    fig_embauche = px.bar(embauche_donnees, x='Annee', y='Total', color='Attrition', 
                          barmode='stack', title="Tendances d'embauche des employés")
    st.plotly_chart(fig_embauche)

    st.subheader("Employés actifs par département")
    employes_par_dept = donnees_employes[donnees_employes['Attrition'] == "No"] \
        .groupby('Department').size().reset_index(name='Nombre')
    fig_dept = px.bar(employes_par_dept, x='Nombre', y='Department', 
                      orientation='h', title="Employés actifs par département")
    st.plotly_chart(fig_dept)

# **Onglet 2 : Démographie**
with onglet2:
    st.header("Démographie")
    col1, col2 = st.columns(2)
    col1.metric("Âge le plus jeune", plus_jeune)
    col2.metric("Âge le plus âgé", plus_age)

    st.subheader("Employés par groupe d'âge")
    groupe_age_counts = donnees_employes['GroupeAge'].value_counts().reset_index()
    groupe_age_counts.columns = ['Groupe d\'âge', 'Nombre']
    fig_groupe_age = px.bar(groupe_age_counts, x='Groupe d\'âge', y='Nombre', title="Employés par groupe d'âge")
    st.plotly_chart(fig_groupe_age)

    st.subheader("Employés par statut marital")
    statut_marital_counts = donnees_employes['MaritalStatus'].value_counts().reset_index()
    statut_marital_counts.columns = ['Statut marital', 'Nombre']
    fig_statut_marital = px.pie(statut_marital_counts, names='Statut marital', values='Nombre', 
                                title="Employés par statut marital")
    st.plotly_chart(fig_statut_marital)

# **Onglet 3 : Suivi de performance**
with onglet3:
    st.header("Suivi de performance")
    noms_employes = donnees_employes['NomComplet'].unique()
    employe_selectionne = st.selectbox("Sélectionnez un employé", noms_employes)

    # Filtrer les données pour l'employé sélectionné
    donnees_employe = donnees_employes[donnees_employes['NomComplet'] == employe_selectionne]
    eval_employe = evaluation_performance[
        evaluation_performance['EmployeeID'] == donnees_employe['EmployeeID'].values[0]
    ]

    st.subheader("Dates clés")
    col1, col2, col3 = st.columns(3)
    col1.metric("Date d'embauche", str(donnees_employe['HireDate'].values[0]))
    derniere_eval = eval_employe['ReviewDate'].max()
    col2.metric("Dernière évaluation", str(derniere_eval))
    prochaine_eval = pd.to_datetime(derniere_eval) + pd.DateOffset(years=1)
    col3.metric("Prochaine évaluation", str(prochaine_eval))

    st.subheader("Évaluations de satisfaction")
    fig_satisfaction = px.line(eval_employe, x='ReviewDate', y='JobSatisfaction', 
                                title="Satisfaction au travail au fil du temps")
    st.plotly_chart(fig_satisfaction)

# **Onglet 4 : Attrition**
with onglet4:
    st.header("Analyse de l'attrition")
    st.metric("Taux d'attrition global", f"{taux_attrition}%")

    st.subheader("Attrition par département et poste")
    attrition_par_dept_job = donnees_employes.groupby(['Department', 'JobRole']) \
        .agg(Total=('EmployeeID', 'count'), 
             Attrition=('Attrition', lambda x: (x == "Yes").sum())) \
        .reset_index()
    attrition_par_dept_job['Taux d\'attrition'] = (attrition_par_dept_job['Attrition'] / 
                                                   attrition_par_dept_job['Total']) * 100
    fig_attrition = px.bar(attrition_par_dept_job, x='Taux d\'attrition', y='Department', color='JobRole', 
                           title="Taux d'attrition par département et poste", orientation='h')
    st.plotly_chart(fig_attrition)

    st.subheader("Attrition par ancienneté")
    attrition_par_anciennete = donnees_employes.groupby('YearsAtCompany') \
        .agg(Total=('EmployeeID', 'count'), 
             Attrition=('Attrition', lambda x: (x == "Yes").sum())) \
        .reset_index()
    attrition_par_anciennete['Taux d\'attrition'] = (attrition_par_anciennete['Attrition'] / 
                                                     attrition_par_anciennete['Total']) * 100
    fig_anciennete = px.line(attrition_par_anciennete, x='YearsAtCompany', y='Taux d\'attrition', 
                             title="Taux d'attrition par ancienneté")
    st.plotly_chart(fig_anciennete)
