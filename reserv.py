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

# Page de commande à emporter
def page_emporter():
    st.title("Commande à Emporter")
    
    name = st.text_input("Nom")
    phone = st.text_input("Numéro de téléphone")
    date = st.date_input("Date de réservation", min_value=datetime(2024, 1, 1), max_value=datetime(2024, 12, 31))
    time = st.time_input("Heure")
    pizza_count = st.number_input("Nombre de pizzas", min_value=1)
    pizzas = [st.text_input(f"Nom de la pizza {i+1}") for i in range(pizza_count)]
    
    if st.button("Ajouter la réservation"):
        reservation = {
            "type": "emporter",
            "name": name,
            "phone": phone,
            "date": date,
            "time": time,
            "pizza_count": pizza_count,
            "pizzas": pizzas
        }
        add_reservation(reservation)
        st.success("Réservation ajoutée avec succès !")
        # Générer le PDF après l'ajout de la réservation
        pdf_file = generate_pdf(st.session_state.reservations)
        st.download_button(
            label="Télécharger le PDF des réservations",
            data=pdf_file,
            file_name="reservations.pdf",
            mime="application/pdf"
        )

# Page de commande sur place
def page_sur_place():
    st.title("Commande sur Place")
    
    name = st.text_input("Nom")
    date = st.date_input("Date de réservation", min_value=datetime(2024, 1, 1), max_value=datetime(2024, 12, 31))
    time = st.time_input("Heure")
    person_count = st.number_input("Nombre de personnes", min_value=1)
    
    if st.button("Ajouter la réservation"):
        reservation = {
            "type": "sur place",
            "name": name,
            "date": date,
            "time": time,
            "person_count": person_count
        }
        add_reservation(reservation)
        st.success("Réservation ajoutée avec succès !")
        # Générer le PDF après l'ajout de la réservation
        pdf_file = generate_pdf(st.session_state.reservations)
        st.download_button(
            label="Télécharger le PDF des réservations",
            data=pdf_file,
            file_name="reservations.pdf",
            mime="application/pdf"
        )

# Fonction pour sauvegarder les réservations dans un fichier PDF
def save_reservations_to_file(reservations):
    pdf = FPDF()
    pdf.add_page()
    
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Liste des Réservations", ln=True, align="C")
    
    sorted_reservations = sorted(reservations, key=lambda x: (x["date"], x["time"]))
    
    emporter_reservations = [r for r in sorted_reservations if r['type'] == 'emporter']
    sur_place_reservations = [r for r in sorted_reservations if r['type'] == 'sur place']

    max_rows = max(len(emporter_reservations), len(sur_place_reservations))
    
    pdf.set_font("Arial", size=10)
    pdf.set_fill_color(200, 200, 200)
    pdf.cell(95, 10, "Emporter", 1, 0, 'C', 1)
    pdf.cell(5, 10, "", 0, 0, 'C', 0) # For the vertical line
    pdf.cell(95, 10, "Sur Place", 1, 1, 'C', 1)
    
    for i in range(max_rows):
        pdf.set_fill_color(255, 255, 255)
        
        # Emporter reservation
        if i < len(emporter_reservations):
            reservation = emporter_reservations[i]
            line = f"{reservation['date']} - {reservation['time']} - {reservation['name']} - {reservation['phone']} - {reservation['pizza_count']} pizzas: {', '.join(reservation['pizzas'])}"
            x, y = pdf.get_x(), pdf.get_y()
            pdf.multi_cell(95, 10, line, 1, 'L', fill=True)
            pdf.set_xy(x + 100, y)  # Move to the right column
        else:
            x, y = pdf.get_x(), pdf.get_y()
            pdf.multi_cell(95, 10, "", 1, 'L', fill=True)
            pdf.set_xy(x + 100, y)  # Move to the right column
        
        # Sur place reservation
        if i < len(sur_place_reservations):
            reservation = sur_place_reservations[i]
            line = f"{reservation['date']} - {reservation['time']} - {reservation['name']} - {reservation['person_count']} personnes"
            x, y = pdf.get_x(), pdf.get_y()
            pdf.multi_cell(95, 10, line, 1, 'L', fill=True)
        else:
            pdf.multi_cell(95, 10, "", 1, 'L', fill=True)
        
        pdf.ln(10)
    
    # Sauvegarder le PDF dans un fichier
    pdf_output = "reservations.pdf"
    pdf.output(pdf_output)

# Fonction pour générer le PDF des réservations
def generate_pdf(reservations):
    save_reservations_to_file(reservations)
    # Lire le contenu du fichier PDF
    with open("reservations.pdf", "rb") as file:
        pdf_data = file.read()
    
    return pdf_data

# Sélection de la page
page = st.sidebar.selectbox("Choisissez une page", ["Commande à Emporter", "Commande sur Place"])

if page == "Commande à Emporter":
    page_emporter()
elif page == "Commande sur Place":
    page_sur_place()