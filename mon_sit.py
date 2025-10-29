import streamlit as st

# Configuration de la page
st.set_page_config(page_title="Data Workers", layout="wide")

# CSS global
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
    </style>
""", unsafe_allow_html=True)

# Navigation
page = st.sidebar.radio("Navigation", ["Les services que je propose", "À propos de moi", "Mes projets"])

# ---------------------------------------------------------------------
# SECTION 1 : SERVICES
# ---------------------------------------------------------------------
if page == "Les services que je propose":
    col1, col2, col3 = st.columns([1, 6, 1])
    with col1:
        st.image("logogo.JPG", caption="AI & ClairData Solutions", use_container_width=True)
    with col2:
        # Animation fade-in + machine à écrire cyclique
        st.markdown("""
            <style>
            .fade-in {
                opacity: 0;
                animation: fadeIn 2s ease-in forwards;
            }
            @keyframes fadeIn {
                from { opacity: 0; transform: translateY(-10px); }
                to { opacity: 1; transform: translateY(0); }
            }
            .main-title {
                text-align: center;
                color: #333;
                font-size: 30px;
                font-weight: 700;
                margin-bottom: 5px;
            }
            .typewriter-container {
                width: 100%;
                text-align: center;
                font-size: 22px;
                font-weight: 600;
                color: #f25287;
                margin-top: 15px;
                height: 35px;
            }
            .typewriter-text {
                display: inline-block;
                border-right: 3px solid #f25287;
                white-space: nowrap;
                overflow: hidden;
                animation: typing 3s steps(40, end), blink .8s step-end infinite;
            }
            @keyframes typing {
                from { width: 0; }
                to { width: 100%; }
            }
            @keyframes blink {
                50% { border-color: transparent; }
            }
            </style>

            <div class="fade-in">
                <h1 class="main-title">Data Scientist, Consultant, Business Analyst & Full-Stack AI Developer</h1>
                <div class="typewriter-container">
                    <span id="typewriter" class="typewriter-text"></span>
                </div>
            </div>

            <script>
            const texts = [
                "Data Science 💡",
                "Développement Web 🌐",
                "Intelligence Artificielle 🤖",
                "Automatisation & Analyse de données 📊"
            ];
            let index = 0;
            let charIndex = 0;
            let currentText = "";
            let isDeleting = false;
            const element = document.getElementById("typewriter");

            function type() {
                const fullText = texts[index];
                if (isDeleting) {
                    currentText = fullText.substring(0, charIndex--);
                } else {
                    currentText = fullText.substring(0, charIndex++);
                }
                element.textContent = currentText;

                if (!isDeleting && charIndex === fullText.length) {
                    setTimeout(() => (isDeleting = true), 1000);
                } else if (isDeleting && charIndex === 0) {
                    isDeleting = false;
                    index = (index + 1) % texts.length;
                }
                setTimeout(type, isDeleting ? 60 : 120);
            }
            window.addEventListener('load', type);
            </script>
        """, unsafe_allow_html=True)
    with col3:
        st.image("dv_lottery.jpg", caption="Alioune Gaye", use_container_width=True)

    st.markdown("""
    Je conçois et déploie des **solutions data et web intelligentes** alliant **analyse de données**, 
    **intelligence artificielle**, **automatisation** et **développement full-stack** pour accompagner la transformation numérique des entreprises.
    """)

    st.markdown("### Mes domaines d’expertise")

    # Ligne 1 : Data Science
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""<div class="service-box"><div class="emoji">📈</div><h3>Analyse exploratoire</h3><p>Nettoyage, structuration et analyse descriptive pour révéler les insights cachés dans vos données.</p></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""<div class="service-box"><div class="emoji">🤖</div><h3>Modélisation prédictive & IA</h3><p>Création de modèles Machine Learning et Deep Learning pour prédire et automatiser la prise de décision.</p></div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""<div class="service-box"><div class="emoji">⚙️</div><h3>Automatisation & API intelligentes</h3><p>Développement d’API combinant OCR, LLM et intégrations cloud pour traiter et structurer des documents complexes.</p></div>""", unsafe_allow_html=True)

    # Ligne 2 : Développement full-stack
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""<div class="service-box"><div class="emoji">💻</div><h3>Développement backend</h3><p>Création d’API robustes avec Node.js, Express, TypeScript et PostgreSQL. Authentification, WebSocket, et automatisation (cron jobs).</p></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""<div class="service-box"><div class="emoji">🎨</div><h3>Frontend moderne</h3><p>Interfaces interactives et élégantes avec React 18, Tailwind CSS, shadcn et Radix UI. Expériences utilisateur fluides et performantes.</p></div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""<div class="service-box"><div class="emoji">🧩</div><h3>Intégrations et connecteurs</h3><p>Connexion à des services tiers : Google APIs, OAuth, SendGrid, AWS S3, IMAP/SMTP, pour des applications interconnectées.</p></div>""", unsafe_allow_html=True)

    # Ligne 3 : Data et formation
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""<div class="service-box"><div class="emoji">📊</div><h3>Tableaux de bord & analytique</h3><p>Création de dashboards dynamiques (Power BI, Streamlit, Shiny, React) pour le suivi en temps réel de vos indicateurs.</p></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""<div class="service-box"><div class="emoji">🗂️</div><h3>Gestion de données</h3><p>Architecture et gestion de bases SQL/NoSQL. Optimisation des requêtes et structuration de vos systèmes d’information.</p></div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""<div class="service-box"><div class="emoji">🎓</div><h3>Formation & accompagnement</h3><p>Sessions de formation personnalisées en Data Science, IA, automatisation et développement web.</p></div>""", unsafe_allow_html=True)

    # Réalisations récentes
    st.markdown("### 🚀 Réalisations récentes intégrées à mes services")
    st.markdown("""
    - **CRM Synergie Marketing Group** : système complet de gestion clients, ventes et commissions (Node.js, React, PostgreSQL, WebSocket).  
    - **API OCR & LLM pour l’immobilier** : extraction automatique et validation de documents (CNI, contrats, bulletins).  
    - **Agent IA juridique** : assistant intelligent multilingue (français / arabe) basé sur un pipeline RAG et embeddings OpenAI.  
    """)

# ---------------------------------------------------------------------
# SECTION 2 : À PROPOS
# ---------------------------------------------------------------------
elif page == "À propos de moi":
    # (identique à ta version précédente)
    st.markdown("<h1 style='text-align: center; color: #333;'>À propos de moi</h1>", unsafe_allow_html=True)
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image("dv_lottery.jpg", use_container_width=True)
    with col2:
        st.markdown("""
        Analyste de données expérimenté et statisticien, je conçois des solutions analytiques et prédictives pour les entreprises.  
        Mon profil combine expertise technique, rigueur scientifique et vision orientée métier.
        """)
    st.markdown("<h2 style='text-align: center; color: #f25287;'>Éducation</h2>", unsafe_allow_html=True)
    st.markdown("**Master en Statistique, Modélisation et Science des données** – Université Claude Bernard Lyon 1.")

    # Autres sections (compétences, techniques, etc.)
    # ... (tu peux garder ton code existant ici)

# ---------------------------------------------------------------------
# SECTION 3 : MES PROJETS
# ---------------------------------------------------------------------
elif page == "Mes projets":
    # (garde le code de ta section projets, déjà enrichie)
    st.markdown("<h1 style='text-align: center; color: #333;'>Mes Projets</h1>", unsafe_allow_html=True)
    # ... (tes projets existants)

# ---------------------------------------------------------------------
# PIED DE PAGE
# ---------------------------------------------------------------------
st.markdown("---")
st.markdown("""
<p style='text-align: center;'>
    Mes contacts :<br>
    <a href='https://www.linkedin.com/in/alioune-gaye-1a5161172/' target='_blank' style='margin-right: 15px;'>
        <img src='https://upload.wikimedia.org/wikipedia/commons/c/ca/LinkedIn_logo_initials.png' style='width:20px;'> LinkedIn
    </a>
    <a href='tel:+33763556982' style='margin-right: 15px;'>
        <img src='https://upload.wikimedia.org/wikipedia/commons/6/6c/Phone_icon.png' style='width:20px;'> 0763556982
    </a>
    <a href='mailto:aliounegaye911@gmail.com'>
        <img src='https://upload.wikimedia.org/wikipedia/commons/2/27/Android_Email_4.4_Icon.png' style='width:20px;'> aliounegaye911@gmail.com
    </a><br>
    © 2025 Data Workers – GAYE ALIOUNE.
</p>
""", unsafe_allow_html=True)
