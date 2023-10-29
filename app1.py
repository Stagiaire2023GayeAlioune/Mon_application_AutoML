import streamlit as st

import numpy as np

import pandas as pd

import pandas_profiling

from streamlit_pandas_profiling import st_profile_report



@st.cache_data

def main():
    st.markdown('<h1 style="text-align: center;">Emploie du temps</h1>', unsafe_allow_html=True)
    col3,col4=st.sidebar.columns(2)
    #st.sidebar.write("<p style='text-align: center;'> Sokhna Faty Bousso : Stagiaire ILM (%s)</p>" % url, unsafe_allow_html=True)
    st.sidebar.write("<p style='text-align: center;'>Automatisation de l'emploi du temps de ma princesse</p>", unsafe_allow_html=True)
    methode1=st.selectbox("selectionner l'heure que tu termine les cours demain",["Week-end","8","9","10","11","12","13","14","15","16","17","18"])
    methode2=st.selectbox("Est ce que tu iras au boulot demain?",["Oui","Non"])
    if methode1=="8" or "9" or "10" or "11" or "12" or "13" or "14" or "15" or "16" or "17":
        if methode2=="Non":
            st.write("Ma cherie, demain tu va rester à la BU pour apprendre les cours, TD ou Tp que tu aura aprés demain et tu rentrera a la maison vers 20h merci:")
        if methode2=="Oui":  
            st.write("Demain Tu rentre aprés les cours pour aller au boulot. Bon courage et prend bien soin de toi merci") 
    if methode1=="Week-end":
        if methode2=="Non":
           st.write("Demain essaye de te reveiller le plus tot possible pour apprendre les cours, TD ou Tp du lundi si possible du mardi aussi ") 
        if methode2=="Oui":
            st.write("Demain, Tu pourras te reposer à condition que tu travail le matin et le soir")
    if methode1=="18":
        st.write("Ma princesse, demain tu rentre à la maison aprés les cours")        
            
               
        #if st.button(" les performances du modèle "):    
if __name__ == "__main__":
    main()    
    
