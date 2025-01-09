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

# Fonction pour sauvegarder les réservations dans un fichier CSV
def save_reservations_to_file(reservations):
    df = pd.DataFrame(reservations)
    df.to_csv("reservations.csv", index=False)

# Fonction pour charger les réservations depuis un fichier
def load_reservations():
    if os.path.exists("reservations.csv"):
        df = pd.read_csv("reservations.csv")
        # Convertir les colonnes `date` et `time` en types datetime
        df['date'] = pd.to_datetime(df['date']).dt.date
        df['time'] = pd.to_datetime(df['time'], format='%H:%M:%S').dt.time
        return df.sort_values(by=["date", "time"]).to_dict('records')  # Trier par date et heure
    return []

# Recharger les données au démarrage
st.session_state.reservations = load_reservations()

# Fonction pour normaliser le numéro de téléphone au format XX XX XX XX XX
def format_phone_number(phone):
    phone = ''.join(filter(str.isdigit, phone))
    if len(phone) == 10:
        return f"{phone[:2]} {phone[2:4]} {phone[4:6]} {phone[6:8]} {phone[8:]}"
    return phone

# Fonction pour modifier une réservation
def modify_reservation(index):
    reservation = st.session_state.reservations[index]
    with st.form(key=f"edit_form_{index}"):
        new_name = st.text_input("Nom", value=reservation["name"])
        new_phone = st.text_input("Numéro de téléphone", value=reservation["phone"])
        new_date = st.date_input("Date de réservation", value=reservation["date"], min_value=datetime(2025, 1, 1), max_value=datetime(2025, 12, 31))
        new_time = st.time_input("Heure", value=reservation["time"])
        new_person_count = st.number_input("Nombre de personnes", min_value=1, value=reservation["person_count"])
        if st.form_submit_button("Enregistrer les modifications"):
            st.session_state.reservations[index] = {
                "type": "sur place",
                "name": new_name,
                "phone": format_phone_number(new_phone),
                "date": str(new_date),
                "time": str(new_time),
                "person_count": new_person_count
            }
            save_reservations_to_file(st.session_state.reservations)
            st.success("Réservation modifiée avec succès !")
            st.experimental_rerun()

# Fonction pour supprimer une réservation
def delete_reservation(index):
    del st.session_state.reservations[index]
    save_reservations_to_file(st.session_state.reservations)
    st.success("Réservation supprimée avec succès !")
    st.experimental_rerun()

# Fonction pour générer un PDF des réservations
def generate_pdf(reservations, title="Liste des Réservations"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=title, ln=True, align="C")
    
    pdf.set_font("Arial", size=10)
    pdf.set_fill_color(200, 200, 200)
    pdf.cell(40, 10, "Nom", 1, 0, 'C', 1)
    pdf.cell(50, 10, "Téléphone", 1, 0, 'C', 1)
    pdf.cell(50, 10, "Date", 1, 0, 'C', 1)
    pdf.cell(30, 10, "Heure", 1, 0, 'C', 1)
    pdf.cell(20, 10, "Np", 1, 1, 'C', 1)
    
    pdf.set_fill_color(255, 255, 255)
    for reservation in reservations:
        pdf.cell(40, 10, str(reservation['name']), 1, 0, 'C', 1)
        pdf.cell(50, 10, str(reservation['phone']), 1, 0, 'C', 1)
        pdf.cell(50, 10, str(reservation['date']), 1, 0, 'C', 1)
        pdf.cell(30, 10, str(reservation['time']), 1, 0, 'C', 1)
        pdf.cell(20, 10, str(reservation['person_count']), 1, 1, 'C', 1)
    
    pdf_file = "reservations.pdf"
    pdf.output(pdf_file)
    return pdf_file

# Page "Commande sur Place"
def page_sur_place():
    st.title("Commande sur Place")
    
    name = st.text_input("Nom")
    phone = st.text_input("Numéro de téléphone")
    date = st.date_input("Date de réservation", min_value=datetime(2025, 1, 1), max_value=datetime(2025, 12, 31))
    time = st.time_input("Heure")
    person_count = st.number_input("Nombre de personnes", min_value=1)
    
    if st.button("Ajouter la réservation"):
        reservation = {
            "type": "sur place",
            "name": name,
            "phone": format_phone_number(phone),
            "date": str(date),
            "time": str(time),
            "person_count": person_count
        }
        add_reservation(reservation)
        st.success("Réservation ajoutée avec succès !")
    
    st.subheader("Réservations existantes")
    for i, reservation in enumerate(st.session_state.reservations):
        col1, col2, col3 = st.columns([4, 1, 1])
        col1.write(f"{reservation['date']} {reservation['time']} - {reservation['name']} - {reservation['phone']} - {reservation['person_count']} personnes")
        col2.button("Modifier", key=f"modify_{i}", on_click=modify_reservation, args=(i,))
        col3.button("Supprimer", key=f"delete_{i}", on_click=delete_reservation, args=(i,))
    
    # Générer un PDF des réservations
    pdf_file = generate_pdf(st.session_state.reservations)
    with open(pdf_file, "rb") as file:
        st.download_button(
            label="Télécharger les réservations en PDF",
            data=file,
            file_name=pdf_file,
            mime="application/pdf"
        )

# Page "Archives"
def page_archives():
    st.title("Archives des Réservations")
    
    if os.path.exists("reservations.csv"):
        df = pd.read_csv("reservations.csv")
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(by=["date", "time"])
        
        months = df['date'].dt.to_period("M").unique()
        for month in months:
            month_df = df[df['date'].dt.to_period("M") == month]
            st.subheader(f"Réservations pour {month}")
            st.dataframe(month_df)
            
            # Télécharger les réservations en PDF
            month_pdf = generate_pdf(month_df.to_dict('records'), title=f"Réservations pour {month}")
            with open(month_pdf, "rb") as file:
                st.download_button(
                    label=f"Télécharger les réservations pour {month}",
                    data=file,
                    file_name=f"reservations_{month}.pdf",
                    mime="application/pdf"
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
