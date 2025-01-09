import streamlit as st
import pandas as pd
from collections import defaultdict
import unicodedata
from fpdf import FPDF

# Fonction pour normaliser une chaîne (supprime les accents, met en minuscule, etc.)
def normalize_string(s):
    if isinstance(s, str):
        s = unicodedata.normalize('NFKD', s).encode('ASCII', 'ignore').decode('utf-8').lower()
    return s

# Fonction pour lire un fichier CSV ou PDF
def read_file(file):
    if file.name.endswith(".csv"):
        df = pd.read_csv(file)
        return df.iloc[1:]  # Ignorer la première ligne
    elif file.name.endswith(".pdf"):
        try:
            from PyPDF2 import PdfReader
            reader = PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
            # Convertir le texte en DataFrame
            lines = text.split('\n')
            data = [line.split() for line in lines if line.strip()]
            df = pd.DataFrame(data)
            return df.iloc[1:]  # Ignorer la première ligne
        except ImportError:
            st.error("Installez PyPDF2 pour traiter les fichiers PDF.")
            return None
    else:
        st.error("Format de fichier non pris en charge. Veuillez utiliser un fichier CSV ou PDF.")
        return None

# Fonction pour filtrer les données
def filter_total_rows(df):
    # Supprimer les lignes contenant "Total" dans n'importe quelle colonne
    df_filtered = df[~df.apply(lambda row: row.astype(str).apply(normalize_string).str.contains("total", na=False).any(), axis=1)]
    return df_filtered

# Fonction pour fusionner les colonnes et renommer
def process_columns(df):
    if len(df.columns) < 3:
        st.error("Le fichier doit contenir au moins trois colonnes.")
        return None
    # Fusionner les deux premières colonnes en "Nom"
    df['Nom'] = df.iloc[:, 0] + " " + df.iloc[:, 1]
    # Renommer les colonnes
    df.rename(columns={df.columns[2]: 'Téléphone', df.columns[3]: 'Date'}, inplace=True)
    # Réorganiser les colonnes
    df = df[['Nom', 'Téléphone', 'Date']]
    # Supprimer la première ligne
    df = df.iloc[1:]
    return df

# Fonction pour traiter les données
def process_data(df):
    if 'Téléphone' not in df.columns:
        st.error("La colonne 'Téléphone' est introuvable dans le fichier.")
        return None

    # Normaliser la colonne "Téléphone"
    df['Téléphone'] = df['Téléphone'].apply(normalize_string)

    phone_counts = defaultdict(int)
    results = []

    for _, row in df.iterrows():
        phone = row['Téléphone']
        if phone not in phone_counts:
            # Compter le nombre d'occurrences
            count = df['Téléphone'].value_counts().get(phone, 0)
            phone_counts[phone] = count
            results.append({'Nom': row['Nom'], 'Téléphone': phone, 'Nombre': count})

    return pd.DataFrame(results)

# Fonction pour générer un PDF
def generate_pdf(data):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Résultats des Téléphones", ln=True, align="C")
    
    # Ajouter les colonnes
    pdf.set_font("Arial", size=10)
    pdf.cell(70, 10, "Nom", 1, 0, 'C')
    pdf.cell(50, 10, "Téléphone", 1, 0, 'C')
    pdf.cell(30, 10, "Nombre", 1, 1, 'C')
    
    # Ajouter les lignes
    for _, row in data.iterrows():
        pdf.cell(70, 10, row['Nom'], 1, 0, 'L')
        pdf.cell(50, 10, row['Téléphone'], 1, 0, 'L')
        pdf.cell(30, 10, str(row['Nombre']), 1, 1, 'C')
    
    # Sauvegarder dans un fichier temporaire
    pdf_output = "resultats_telephones.pdf"
    pdf.output(pdf_output)
    return pdf_output

# Interface utilisateur Streamlit
st.title("Analyse des Téléphones dans un Fichier")

# Initialiser les réservations dans la session
if 'reservations' not in st.session_state:
    st.session_state.reservations = []

# Demander à l'utilisateur de fournir un fichier
uploaded_file = st.file_uploader("Chargez un fichier CSV ou PDF", type=["csv", "pdf"])

if uploaded_file:
    st.write("Fichier chargé avec succès !")
    
    # Lecture du fichier
    df = read_file(uploaded_file)
    if df is not None:
        # Filtrer les lignes contenant "Total"
        df_filtered = filter_total_rows(df)
        
        # Modifier les colonnes (fusion et renommage)
        df_modified = process_columns(df_filtered)
        if df_modified is not None:
            st.write("Aperçu des données filtrées et modifiées :")
            st.write(df_modified)  # Affiche le tableau complet
            
            # Traitement des données
            results = process_data(df_modified)
            if results is not None:
                st.write("Résultats :")
                st.dataframe(results)  # Affiche les résultats sous forme de tableau
                
                # Télécharger les résultats en format tableau sans index
                csv = results.to_csv(index=False, sep=',').encode('utf-8')
                st.download_button(
                    label="Télécharger les résultats en CSV",
                    data=csv,
                    file_name='resultats_telephones.csv',
                    mime='text/csv',
                )

                # Télécharger les résultats en PDF
                pdf_file = generate_pdf(results)
                with open(pdf_file, "rb") as file:
                    st.download_button(
                        label="Télécharger les résultats en PDF",
                        data=file,
                        file_name='resultats_telephones.pdf',
                        mime='application/pdf',
                    )
