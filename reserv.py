import streamlit as st
from datetime import datetime
from fpdf import FPDF
import os

# Initialiser les données de réservation
if 'reservations' not in st.session_state:
    st.session_state.reservations = []

# Fonction pour ajouter une réservation
def add_reservation(reservation):
    st.session_state.reservations.append(reservation)
    save_reservations_to_file(st.session_state.reservations)

# Calcul du nombre total de personnes pour chaque date
def total_person_count_by_date():
    person_count_by_date = {}
    for reservation in st.session_state.reservations:
        date = reservation['date']
        person_count_by_date[date] = person_count_by_date.get(date, 0) + reservation['person_count']
    return person_count_by_date

# Fonction pour supprimer une réservation
def delete_reservation(index):
    del st.session_state.reservations[index]
    save_reservations_to_file(st.session_state.reservations)
    st.success("Réservation supprimée avec succès !")

# Fonction pour modifier une réservation
def edit_reservation(index):
    reservation = st.session_state.reservations[index]
    with st.form(key=f"edit_form_{index}", clear_on_submit=True):
        new_name = st.text_input("Nom", value=reservation["name"])
        new_phone = st.text_input("Numéro de téléphone", value=reservation["phone"])
        new_date = st.date_input("Date de réservation", value=reservation["date"])
        new_time = st.time_input("Heure", value=reservation["time"])
        new_person_count = st.number_input("Nombre de personnes", min_value=1, value=reservation["person_count"])
        
        if st.form_submit_button("Confirmer la modification"):
            st.session_state.reservations[index] = {
                "type": "sur place",
                "name": new_name,
                "phone": new_phone,
                "date": new_date,
                "time": new_time,
                "person_count": new_person_count
            }
            save_reservations_to_file(st.session_state.reservations)
            st.success("Réservation modifiée avec succès !")

# Page de commande sur place
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
            "date": date,
            "time": time,
            "person_count": person_count
        }
        add_reservation(reservation)
        st.success("Réservation ajoutée avec succès !")
        
        # Afficher le nombre total de personnes pour les réservations du même jour
        total_count = total_person_count_by_date().get(date, 0)
        st.write(f"Nombre total de personnes pour les réservations du {date}: {total_count}")

        # Générer le PDF après l'ajout de la réservation
        pdf_file = generate_pdf(st.session_state.reservations)
        st.download_button(
            label="Télécharger le PDF des réservations",
            data=pdf_file,
            file_name="reservations.pdf",
            mime="application/pdf"
        )
    
    st.subheader("Réservations existantes")
    for i, reservation in enumerate(st.session_state.reservations):
        # Afficher les informations de la réservation dans une ligne
        col1, col2, col3, col4 = st.columns([2, 2, 1, 1])
        
        with col1:
            st.write(f"{reservation['date']} - {reservation['time']} - {reservation['name']} - {reservation['phone']} - {reservation['person_count']} personnes")
        
        with col3:
            if st.button("Modifier", key=f"edit_btn_{i}"):
                edit_reservation(i)
        
        with col4:
            if st.button("Supprimer", key=f"delete_btn_{i}"):
                delete_reservation(i)
                st.experimental_rerun()  # Rafraîchir la page après suppression

# Fonction pour sauvegarder les réservations dans un fichier PDF
def save_reservations_to_file(reservations):
    pdf = FPDF()
    pdf.add_page()
    
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Liste des Réservations - Semaine", ln=True, align="C")
    
    sorted_reservations = sorted(reservations, key=lambda x: (x["date"], x["time"]))
    person_count_by_date = total_person_count_by_date()
    
    pdf.set_font("Arial", size=10)
    pdf.set_fill_color(200, 200, 200)
    pdf.cell(40, 10, "Nom", 1, 0, 'C', 1)
    pdf.cell(40, 10, "Téléphone", 1, 0, 'C', 1)
    pdf.cell(50, 10, "Date et Heure", 1, 0, 'C', 1)
    pdf.cell(30, 10, "Np", 1, 1, 'C', 1)
    
    current_date = None
    pdf.set_fill_color(255, 255, 255)
    for reservation in sorted_reservations:
        # Vérifier si la date change pour ajouter le total des personnes
        if current_date and reservation['date'] != current_date:
            pdf.set_font("Arial", 'B', 10)
            pdf.cell(130, 10, f"Total pour le {current_date}:", 1, 0, 'R', fill=True)
            pdf.cell(30, 10, str(person_count_by_date[current_date]), 1, 1, 'C', fill=True)
            pdf.set_font("Arial", size=10)
        
        # Sauvegarder la date actuelle
        current_date = reservation['date']
        
        # Ajouter les détails de la réservation sur une seule ligne
        line_name = reservation['name']
        line_phone = reservation['phone']
        line_date_time = f"{reservation['date']} {reservation['time']}"
        line_person_count = str(reservation['person_count'])
        
        pdf.cell(40, 10, line_name, 1, 0, 'L', fill=True)
        pdf.cell(40, 10, line_phone, 1, 0, 'L', fill=True)
        pdf.cell(50, 10, line_date_time, 1, 0, 'L', fill=True)
        pdf.cell(30, 10, line_person_count, 1, 1, 'L', fill=True)
    
    # Ajouter le total des personnes pour la dernière date du fichier
    if current_date:
        pdf.set_font("Arial", 'B', 10)
        pdf.cell(130, 10, f"Total pour le {current_date}:", 1, 0, 'R', fill=True)
        pdf.cell(30, 10, str(person_count_by_date[current_date]), 1, 1, 'C', fill=True)
    
    # Sauvegarder le PDF dans un fichier
    pdf_output = "reservations.pdf"
    pdf.output(pdf_output)

# Fonction pour générer le PDF des réservations
def generate_pdf(reservations):
    save_reservations_to_file(reservations)
    with open("reservations.pdf", "rb") as file:
        pdf_data = file.read()
    
    return pdf_data

# Affichage de la page "Commande sur Place"
page_sur_place()
