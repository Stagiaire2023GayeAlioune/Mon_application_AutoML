import streamlit as st
from datetime import datetime
from fpdf import FPDF
import pandas as pd
import os

# Initialiser les données de réservation
if 'reservations' not in st.session_state:
    st.session_state.reservations = []

# Fonction pour ajouter une réservation
def add_reservation(reservation):
    st.session_state.reservations.append(reservation)
    save_reservations_to_file(st.session_state.reservations)

# Fonction pour sauvegarder les réservations dans un fichier PDF
def save_reservations_to_file(reservations):
    # Sauvegarder dans un fichier CSV pour simplifier l'archivage
    df = pd.DataFrame(reservations)
    df.to_csv("reservations.csv", index=False)

# Fonction pour charger les réservations depuis un fichier
def load_reservations():
    if os.path.exists("reservations.csv"):
        df = pd.read_csv("reservations.csv")
        # Convertir les colonnes `date` et `time` en types datetime
        df['date'] = pd.to_datetime(df['date']).dt.date
        df['time'] = pd.to_datetime(df['time'], format='%H:%M:%S').dt.time
        return df.to_dict('records')
    return []

# Recharger les données au démarrage
st.session_state.reservations = load_reservations()

# Calcul du nombre total de personnes pour chaque date
def total_person_count_by_date():
    person_count_by_date = {}
    for reservation in st.session_state.reservations:
        date = reservation['date']
        person_count_by_date[date] = person_count_by_date.get(date, 0) + reservation['person_count']
    return person_count_by_date

# Fonction pour afficher la page principale "Commande sur Place"
def page_sur_place():
    st.title("Commande sur Place")
    
    name = st.text_input("Nom")
    phone = st.text_input("Numéro de téléphone")
    date = st.date_input("Date de réservation", min_value=datetime(2024, 1, 1), max_value=datetime(2024, 12, 31))
    time = st.time_input("Heure")
    person_count = st.number_input("Nombre de personnes", min_value=1)
    
    if st.button("Ajouter la réservation"):
        reservation = {
            "type": "sur place",
            "name": name,
            "phone": phone,
            "date": str(date),
            "time": str(time),
            "person_count": person_count
        }
        add_reservation(reservation)
        st.success("Réservation ajoutée avec succès !")
        
        # Afficher le nombre total de personnes pour les réservations du même jour
        total_count = total_person_count_by_date().get(date, 0)
        st.write(f"Nombre total de personnes pour les réservations du {date}: {total_count}")
    
    st.subheader("Réservations existantes")
    for i, reservation in enumerate(st.session_state.reservations):
        st.write(f"{reservation['date']} - {reservation['time']} - {reservation['name']} - {reservation['phone']} - {reservation['person_count']} personnes")

# Fonction pour afficher la page des archives
def page_archives():
    st.title("Archives des Réservations")
    
    # Charger les réservations depuis le fichier CSV
    if os.path.exists("reservations.csv"):
        df = pd.read_csv("reservations.csv")
        # Convertir la colonne 'date' en format datetime
        df['date'] = pd.to_datetime(df['date'])
        df['month'] = df['date'].dt.strftime('%Y-%m')  # Extraire le mois et l'année
        
        # Grouper par mois
        months = df['month'].unique()
        for month in sorted(months):
            st.subheader(f"Réservations pour le mois : {month}")
            monthly_data = df[df['month'] == month][['name', 'phone', 'date', 'time', 'person_count']]
            monthly_data.rename(columns={
                'name': 'Nom',
                'phone': 'Téléphone',
                'date': 'Date',
                'time': 'Heure',
                'person_count': 'Nombre de personnes'
            }, inplace=True)
            st.dataframe(monthly_data)
            
            # Ajouter bouton pour exporter en CSV
            csv_data = monthly_data.to_csv(index=False).encode('utf-8')
            st.download_button(
                label=f"Télécharger les réservations de {month}",
                data=csv_data,
                file_name=f"reservations_{month}.csv",
                mime='text/csv',
            )
    else:
        st.warning("Aucune réservation enregistrée pour le moment.")

# Navigation entre les pages
st.sidebar.title("Menu")
page = st.sidebar.radio("Choisissez une page", ["Commande sur Place", "Archives"])

if page == "Commande sur Place":
    page_sur_place()
elif page == "Archives":
    page_archives()
