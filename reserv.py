import streamlit as st
from datetime import datetime, date, time
from fpdf import FPDF
import os
import json

# Chemin du fichier de sauvegarde des réservations
SAVE_FILE = "reservations.json"

# Charger les réservations depuis le fichier JSON
def load_reservations():
    if os.path.exists(SAVE_FILE):
        with open(SAVE_FILE, "r") as f:
            try:
                reservations = json.load(f)
                # Convertir les chaînes de caractères en objets date et time
                for reservation in reservations:
                    reservation['date'] = datetime.strptime(reservation['date'], "%Y-%m-%d").date()
                    reservation['time'] = datetime.strptime(reservation['time'], "%H:%M:%S").time()
                return reservations
            except json.JSONDecodeError:
                # Si le fichier est vide ou mal formé, retourner une liste vide
                return []
    return []

# Sauvegarder les réservations dans le fichier JSON
def save_reservations():
    # Convertir les objets date et time en chaînes de caractères
    reservations_to_save = [
        {**reservation, 'date': reservation['date'].strftime("%Y-%m-%d"), 'time': reservation['time'].strftime("%H:%M:%S")}
        for reservation in st.session_state.reservations
    ]
    with open(SAVE_FILE, "w") as f:
        json.dump(reservations_to_save, f)

# Initialiser les données de réservation
if 'reservations' not in st.session_state:
    st.session_state.reservations = load_reservations()

# Fonction pour ajouter une réservation
def add_reservation(reservation):
    st.session_state.reservations.append(reservation)
    save_reservations()
    save_reservations_to_file(st.session_state.reservations)

# Fonction pour supprimer une réservation
def delete_reservation(index):
    st.session_state.reservations.pop(index)
    save_reservations()
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
    comments = st.text_area("Commentaires")
    
    if st.button("Ajouter la réservation"):
        reservation = {
            "type": "sur place",
            "name": name,
            "date": date,
            "time": time,
            "person_count": person_count,
            "comments": comments
        }
        add_reservation(reservation)
        st.success("Réservation ajoutée avec succès !")
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
            pdf.set_xy(x + 100, y)  # Move to the right column
        else:
            x, y = pdf.get_x(), pdf.get_y()
            pdf.multi_cell(95, 10, "", 1, 'L', fill=True)
            pdf.set_xy(x + 100, y)  # Move to the right column
        
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

# Fonction pour afficher les réservations et permettre leur suppression
def view_reservations():
    st.title("Visualiser et Supprimer les Réservations")
    
    if len(st.session_state.reservations) == 0:
        st.info("Aucune réservation n'a été effectuée.")
    else:
        for index, reservation in enumerate(st.session_state.reservations):
            if reservation["type"] == "emporter":
                st.write(f"Emporter - {reservation['date']} {reservation['time']} - {reservation['name']} - {reservation['phone']} - {reservation['pizza_count']} pizzas: {', '.join(reservation['pizzas'])}")
            else:
                st.write(f"Sur Place - {reservation['date']} {reservation['time']} - {reservation['name']} - {reservation['person_count']} personnes - {reservation['comments']}")
            if st.button(f"Supprimer la réservation {index+1}", key=f"delete_{index}"):
                delete_reservation(index)
                st.experimental_rerun()

    pdf_file = generate_pdf(st.session_state.reservations)
    st.download_button(
        label="Télécharger le PDF des réservations",
        data=pdf_file,
        file_name="reservations.pdf",
        mime="application/pdf"
    )

# Sélection de la page
page = st.sidebar.selectbox("Choisissez le type de reservation", ["Commande à Emporter", "Commande sur Place", "Visualiser les Réservations"])
st.sidebar.image("casa.PNG")

if page == "Commande à Emporter":
    page_emporter()
elif page == "Commande sur Place":
    page_sur_place()
elif page == "Visualiser les Réservations":
    view_reservations()
