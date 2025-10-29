import streamlit as st

# Configuration de la page
st.set_page_config(page_title="Data Workers", layout="wide")

# CSS personnalisé
st.markdown("""
    <style>
    .header-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background-color: #f9f9f9;
        padding: 10px;
    }
    .header-container img {
        max-height: 150px;
        object-fit: cover;
        border-radius: 10px;
    }
    .service-box, .project-box {
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 20px;
        margin: 10px;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
        background-color: #fff;
        text-align: center;
    }
    .service-box h3, .project-box h3 {
        color: #333;
        margin-bottom: 10px;
    }
    .service-box p, .project-box p {
        color: #666;
        font-size: 14px;
    }
    .service-box .emoji {
        font-size: 40px;
        margin-bottom: 10px;
    }
    .project-box a {
        color: #f25287;
        text-decoration: none;
        font-weight: bold;
    }
    .project-box img {
        max-width: 100%;
        border-radius: 10px;
        margin-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Navigation
page = st.sidebar.radio("Navigation", ["Les services que je propose", "À propos de moi", "Mes projets"])

# ---------------------------------------------------------------------
# SECTION : SERVICES
# ---------------------------------------------------------------------
if page == "Les services que je propose":
    col1, col2, col3 = st.columns([1, 6, 1])
    with col1:
        st.image("logogo.JPG", caption="AI & ClairData Solutions", use_container_width=True)
    with col2:
        st.markdown("<h1 style='text-align: center; color: #333;'>Data Scientist, Consultant, Business Analyst, Shiny et Streamlit Developper</h1>", unsafe_allow_html=True)
        st.markdown("<h2 style='text-align: center; color: #f25287;'>La donnée au service de votre croissance</h2>", unsafe_allow_html=True)
    with col3:
        st.image("dv_lottery.jpg", caption="Alioune Gaye", use_container_width=True)

    st.markdown("Je propose des services de pointe en intelligence artificielle et en science des données. Ma mission est d'accompagner les entreprises dans leur prise de décision en exploitant la puissance des données.")
    st.markdown("### Services")
    
    # Ligne 1
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""<div class="service-box"><div class="emoji">📈</div><h3>Analyse exploratoire</h3><p>Nettoyage, structuration, et analyse descriptive.</p></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""<div class="service-box"><div class="emoji">🤖</div><h3>Modélisation prédictive</h3><p>Algorithmes de Machine Learning et Deep Learning adaptés aux besoins.</p></div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""<div class="service-box"><div class="emoji">⚙️</div><h3>Automatisation</h3><p>Création d'outils pour automatiser des tâches répétitives.</p></div>""", unsafe_allow_html=True)

    # Ligne 2
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""<div class="service-box"><div class="emoji">📊</div><h3>Tableaux de bord</h3><p>Développement de dashboards interactifs pour le suivi des KPI.</p></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""<div class="service-box"><div class="emoji">🗂️</div><h3>Bases de données</h3><p>Gestion et structuration de bases de données.</p></div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""<div class="service-box"><div class="emoji">🎓</div><h3>Formations</h3><p>Formations sur l'IA et les statistiques appliquées.</p></div>""", unsafe_allow_html=True)

# ---------------------------------------------------------------------
# SECTION : À PROPOS DE MOI
# ---------------------------------------------------------------------
elif page == "À propos de moi":
    st.markdown("<h1 style='text-align: center; color: #333;'>À propos de moi</h1>", unsafe_allow_html=True)
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image("dv_lottery.jpg", use_container_width=True)
    with col2:
        st.markdown("""
        Analyste de données expérimenté et statisticien, je suis compétent en gestion de projet, suivi et évaluation, ainsi qu'en gestion de bases de données.
        Je participe à la collecte, au suivi et à l’analyse de données, la création de dashboards et la modélisation statistique.
        """)

    # Éducation
    st.markdown("<h2 style='text-align: center; color: #f25287;'>Éducation</h2>", unsafe_allow_html=True)
    st.markdown("**Statistique, Modélisation et Science des données**, Université Claude Bernard Lyon 1 (Bac +5).")

    # Comportemental
    st.markdown("<h2 style='text-align: center; color: #f25287;'>Comportemental</h2>", unsafe_allow_html=True)
    st.markdown("""
    - Bonne communication orale et écrite, travail en équipe ou indépendant  
    - Compétences organisationnelles, autonomie, innovation, rigueur  
    - Sociable, professionnel, partage de connaissances
    """)

    # Techniques
    st.markdown("<h2 style='text-align: center; color: #f25287;'>Techniques</h2>", unsafe_allow_html=True)
    st.markdown("""
    - **Machine Learning**, **Deep Learning**, **Fouille de données**, **NLP**, **Text mining**, **Vision par ordinateur**  
    - **Statistique**, **Analyse de données**, **Séries temporelles**, **Spectrométrie de masse**  
    - **Cloud (Azure)**, **Git/GitHub**, **Django**, **Scrapy**
    - **Backend :** Node.js, Express, TypeScript  
    - **ORM/BD :** Drizzle ORM, PostgreSQL (connect-pg-simple)  
    - **Auth/session :** express-session, passport-local  
    - **Frontend :** React 18, Vite, TypeScript  
    - **UI :** Radix UI, shadcn, Tailwind CSS  
    - **State/data :** @tanstack/react-query  
    - **WebSocket :** ws (temps réel)  
    - **Email :** Nodemailer (SMTP Hostinger), SendGrid  
    - **Tâches/cron :** node-cron  
    - **Fichiers/Uploads :** multer, Uppy (AWS S3 adapter)  
    - **PDF/Docs :** jspdf, html2pdf.js, docxtemplater, mammoth  
    - **Google :** googleapis, google-auth-library (OAuth, Calendar)  
    - **Divers :** zod, date-fns, sharp
    """)

    # Langages & outils
    st.markdown("<h2 style='text-align: center; color: #f25287;'>Langages de programmation</h2>", unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.image("python_icon.PNG", use_container_width=True)
    with col2:
        st.image("r_icon.PNG", use_container_width=True)
    with col3:
        st.image("stata_icon.PNG", use_container_width=True)
    with col4:
        st.image("C++.PNG", use_container_width=True)

    st.markdown("<h2 style='text-align: center; color: #f25287;'>Visualisation</h2>", unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.image("power_bi_icon.PNG", use_container_width=True)
    with col2:
        st.image("excel_icon.PNG", use_container_width=True)
    with col3:
        st.image("streamlit.PNG", use_container_width=True)
    with col4:
        st.image("shiny.PNG", use_container_width=True)

    st.markdown("<h2 style='text-align: center; color: #f25287;'>Bases de données</h2>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.image("mysql_icon.PNG", use_container_width=True)
    with col2:
        st.image("sql_server_icon.PNG", use_container_width=True)

    st.markdown("<h2 style='text-align: center; color: #f25287;'>Bureautique</h2>", unsafe_allow_html=True)
    st.image("tableau_icon.PNG", use_container_width=True)

# ---------------------------------------------------------------------
# SECTION : MES PROJETS
# ---------------------------------------------------------------------
elif page == "Mes projets":
    st.markdown("<h1 style='text-align: center; color: #333;'>Mes Projets</h1>", unsafe_allow_html=True)

    # Ligne 1
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image("Alzeimer.PNG", caption="Détection Alzheimer", use_container_width=True)
        st.markdown("""
        <div class="project-box">
            <h3>Détection de la Maladie d'Alzheimer</h3>
            <p>Deep Learning (VGG19, ResNet50) pour détecter les stades de démence à partir d’IRM cérébrales.</p>
            <a href="https://view.officeapps.live.com/op/view.aspx?src=https://raw.githubusercontent.com/Stagiaire2023GayeAlioune/Mon_application_AutoML/refs/heads/master/Detection_Alzheimer_Deep_Learning.docx">Télécharger le rapport</a>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.image("cancer.PNG", caption="Cancer du Sein", use_container_width=True)
        st.markdown("""
        <div class="project-box">
            <h3>Détection du Cancer du Sein</h3>
            <p>Classification des masses mammaires échographiques en bénin, malin et normal.</p>
            <a href="https://github.com/Stagiaire2023GayeAlioune/Mon_application_AutoML/blob/master/Rapport_Cancer_du_sein.pdf">Télécharger le rapport</a>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.image("carte.PNG", caption="Fraude Bancaire", use_container_width=True)
        st.markdown("""
        <div class="project-box">
            <h3>Détection de Fraude Bancaire</h3>
            <p>Classification des transactions bancaires pour détecter les fraudes.</p>
            <a href="https://github.com/Stagiaire2023GayeAlioune/Mon_application_AutoML/blob/master/Rapport_detection_fraude.pdf">Télécharger le rapport</a>
        </div>
        """, unsafe_allow_html=True)

    # Ligne 2
    col1, col2 = st.columns(2)
    with col1:
        st.image("credi.jpg", caption="Analyse des Risques de Crédit", use_container_width=True)
        st.markdown("""
        <div class="project-box">
            <h3>Analyse des Risques de Crédit</h3>
            <p>Évaluation des risques de crédit à l’aide du Machine Learning.</p>
            <a href="https://risquedecreditsclients.streamlit.app/">Accéder à l'application</a>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.image("RH.PNG", caption="Tableau de Bord RH", use_container_width=True)
        st.markdown("""
        <div class="project-box">
            <h3>Tableau de Bord RH</h3>
            <p>Dashboard RH interactif (attrition, performance, démographie).</p>
            <a href="https://applicationtableaudebordanalyserh.streamlit.app/">Accéder à l'application</a>
        </div>
        """, unsafe_allow_html=True)

    # Ligne 3 — Nouveaux projets
    col1, col2 = st.columns(2)
    with col1:
        st.image("crm_synergie.png", caption="CRM Synergie Marketing Group", use_container_width=True)
        st.markdown("""
        <div class="project-box">
            <h3>CRM Synergie Marketing Group</h3>
            <p>CRM complet orienté recrutement/vente/MLM. Stack : React, Node.js, Drizzle ORM, PostgreSQL, WebSocket, Tailwind, etc.</p>
            <p><strong>Période :</strong> 10/07/2025 – 25/09/2025</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.image("api_ocr.png", caption="API OCR & LLM", use_container_width=True)
        st.markdown("""
        <div class="project-box">
            <h3>API OCR & LLM pour documents immobiliers</h3>
            <p>Extraction automatique de données structurées à partir de PDF/images grâce à l’OCR et aux modèles de langage (LLM).</p>
        </div>
        """, unsafe_allow_html=True)

    # Ligne 4
    col1, col2 = st.columns(2)
    with col1:
        st.image("so2_cancer.png", caption="SO₂ et Cancer du Poumon", use_container_width=True)
        st.markdown("""
        <div class="project-box">
            <h3>Relation entre le SO₂ et le risque de cancer du poumon</h3>
            <p>Étude statistique sur les travailleurs exposés au dioxyde de soufre.</p>
            <a href="https://github.com/Stagiaire2023GayeAlioune/Mon_application_AutoML/blob/master/Rapport_Complet_Cancer_Poumon_SO2%202.pdf">Télécharger le rapport</a>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.image("developpement_durable.png", caption="Développement Durable", use_container_width=True)
        st.markdown("""
        <div class="project-box">
            <h3>Projet Développement Durable</h3>
            <p>Identification de pistes concrètes pour améliorer la durabilité.</p>
            <a href="https://github.com/Stagiaire2023GayeAlioune/Mon_application_AutoML/blob/master/Description_Projet_Developpement_Durable.docx">Télécharger la description</a>
        </div>
        """, unsafe_allow_html=True)

    # Ligne 5
    st.image("ai_juridique.png", caption="Agent IA Juridique", use_container_width=True)
    st.markdown("""
    <div class="project-box">
        <h3>Agent IA juridique multilingue (Français / Arabe)</h3>
        <p>Pipeline RAG (FAISS + OpenAI embeddings) pour répondre à des questions juridiques à partir de bases documentaires internes (PDF, DOCX, HTML).</p>
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------------------
# PIED DE PAGE
# ---------------------------------------------------------------------
st.markdown("---")
st.markdown("""
<p style='text-align: center;'>
    Mes contacts :<br>
    <a href='https://www.linkedin.com/in/alioune-gaye-1a5161172/' target='_blank' style='margin-right: 15px;'>
        <img src='https://upload.wikimedia.org/wikipedia/commons/c/ca/LinkedIn_logo_initials.png' alt='LinkedIn' style='width:20px; vertical-align:middle;'> LinkedIn
    </a>
    <a href='tel:+33763556982' style='margin-right: 15px;'>
        <img src='https://upload.wikimedia.org/wikipedia/commons/6/6c/Phone_icon.png' alt='Phone' style='width:20px; vertical-align:middle;'> 0763556982
    </a>
    <a href='mailto:aliounegaye911@gmail.com'>
        <img src='https://upload.wikimedia.org/wikipedia/commons/2/27/Android_Email_4.4_Icon.png' alt='Email' style='width:20px; vertical-align:middle;'> aliounegaye911@gmail.com
    </a> <br>
    © 2025 Data Workers – GAYE ALIOUNE.
</p>
""", unsafe_allow_html=True)
