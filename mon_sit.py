import streamlit as st

# -----------------------------------------------------
# CONFIGURATION GLOBALE
# -----------------------------------------------------
st.set_page_config(page_title="Data Workers", page_icon="🧠", layout="wide")

# -----------------------------------------------------
# STYLE GLOBAL
# -----------------------------------------------------
st.markdown("""
    <style>
    * { font-family: 'Poppins', sans-serif; }

    /* Barre de navigation */
    .navbar {
        background-color: #fff;
        padding: 12px 0;
        position: fixed;
        top: 0;
        width: 100%;
        z-index: 999;
        border-bottom: 1px solid #eee;
        text-align: center;
    }
    .navbar a {
        color: #f25287;
        text-decoration: none;
        font-weight: 600;
        margin: 0 18px;
        transition: all 0.3s ease;
    }
    .navbar a:hover {
        color: #ff8ba7;
        text-decoration: underline;
    }

    /* Effet de transition globale */
    .main {
        opacity: 0;
        animation: fadeIn 1.2s ease forwards;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* Hero Section */
    .hero {
        background: linear-gradient(120deg, #f25287, #ff8ba7, #ffd1dc);
        background-size: 300% 300%;
        animation: gradientShift 10s ease infinite;
        padding: 90px 20px 60px;
        color: white;
        text-align: center;
        border-radius: 20px;
        margin-top: 70px;
        margin-bottom: 40px;
    }
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    .hero h1 { font-size: 38px; font-weight: 800; margin-bottom: 10px; }
    .hero h2 { font-size: 20px; font-weight: 500; margin-bottom: 25px; }
    .btn-contact {
        background: white;
        color: #f25287;
        padding: 10px 25px;
        border-radius: 8px;
        text-decoration: none;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .btn-contact:hover { background: #f25287; color: white; }

    /* Services & Projets */
    .service-box, .project-box {
        border: 1px solid #eee;
        border-radius: 12px;
        padding: 20px;
        margin: 10px;
        background-color: #fff;
        text-align: center;
        transition: all 0.3s ease;
        box-shadow: 0 2px 6px rgba(0,0,0,0.05);
    }
    .service-box:hover, .project-box:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 18px rgba(242,82,135,0.25);
    }
    .service-box h3, .project-box h3 { color: #333; margin-bottom: 10px; }
    .emoji { font-size: 40px; margin-bottom: 8px; }

    /* Pied de page */
    footer {
        text-align: center;
        margin-top: 40px;
        border-top: 1px solid #eee;
        padding-top: 15px;
        font-size: 14px;
    }
    </style>

    <nav class="navbar">
        <a href="#services">Services</a>
        <a href="#about">À propos</a>
        <a href="#projects">Projets</a>
        <a href="#contact">Contact</a>
    </nav>
""", unsafe_allow_html=True)

# -----------------------------------------------------
# PAGE 1 : LES SERVICES
# -----------------------------------------------------
st.markdown("""
<div class="hero">
    <h1>🚀 Data Workers</h1>
    <h2>La donnée, le web et l’intelligence artificielle au service de votre croissance</h2>
    <a href="mailto:aliounegaye911@gmail.com" class="btn-contact">📩 Me contacter</a>
</div>
""", unsafe_allow_html=True)

st.markdown("<h2 id='services' style='color:#f25287;'>💼 Les services que je propose</h2>", unsafe_allow_html=True)
st.write("Je conçois et déploie des solutions **data et web intelligentes** alliant IA, automatisation et développement full-stack.")

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("""<div class="service-box"><div class="emoji">📈</div><h3>Analyse exploratoire</h3><p>Nettoyage, structuration et visualisation de données pour révéler des insights décisionnels.</p></div>""", unsafe_allow_html=True)
with col2:
    st.markdown("""<div class="service-box"><div class="emoji">🤖</div><h3>IA & Prédiction</h3><p>Modèles de Machine Learning et Deep Learning pour anticiper les comportements et automatiser les processus.</p></div>""", unsafe_allow_html=True)
with col3:
    st.markdown("""<div class="service-box"><div class="emoji">⚙️</div><h3>API & Automatisation</h3><p>Création d’API intelligentes combinant OCR, LLM et intégrations cloud (Google, AWS, Azure).</p></div>""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("""<div class="service-box"><div class="emoji">💻</div><h3>Développement Backend</h3><p>Node.js, Express, TypeScript, PostgreSQL et Drizzle ORM. Authentification, WebSocket, tâches planifiées.</p></div>""", unsafe_allow_html=True)
with col2:
    st.markdown("""<div class="service-box"><div class="emoji">🎨</div><h3>Frontend Moderne</h3><p>React 18, Tailwind CSS, Radix UI, shadcn — interfaces performantes et esthétiques.</p></div>""", unsafe_allow_html=True)
with col3:
    st.markdown("""<div class="service-box"><div class="emoji">🧩</div><h3>Intégrations</h3><p>Connexion à Google APIs, SendGrid, AWS S3, OAuth, IMAP/SMTP et services tiers.</p></div>""", unsafe_allow_html=True)

# -----------------------------------------------------
# PAGE 2 : À PROPOS
# -----------------------------------------------------
st.markdown("<h2 id='about' style='color:#f25287;'>👤 À propos de moi</h2>", unsafe_allow_html=True)
col1, col2 = st.columns([1, 2])
with col1:
    st.image("dv_lottery.jpg", use_container_width=True)
with col2:
    st.markdown("""
    Je suis **Alioune Gaye**, Data Scientist, statisticien et développeur full-stack spécialisé en IA et automatisation.  
    Mon objectif : transformer la donnée en valeur métier à travers des solutions concrètes, performantes et évolutives.
    """)
st.markdown("**🎓 Éducation :** Master en Statistique, Modélisation et Science des données – Université Claude Bernard Lyon 1 (Bac +5).")
st.markdown("**💡 Compétences comportementales :** communication claire, rigueur, innovation, autonomie, leadership technique.")
st.markdown("**🧠 Compétences techniques :** Machine Learning, Deep Learning, NLP, Vision, Node.js, React, Tailwind CSS, PostgreSQL, Docker, Azure, Django, GitHub.")
st.markdown("**💻 Langages :** Python, R, C++, SQL, TypeScript, JavaScript, Stata.")
st.markdown("**📊 Outils :** Power BI, Excel, Streamlit, Shiny, Tableau.")

# -----------------------------------------------------
# PAGE 3 : PROJETS
# -----------------------------------------------------
st.markdown("<h2 id='projects' style='color:#f25287;'>🧩 Mes projets</h2>", unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)
with col1:
    st.image("Alzeimer.PNG", caption="Détection Alzheimer", use_container_width=True)
    st.markdown("""<div class="project-box"><h3>Détection de la Maladie d'Alzheimer</h3><p>Deep Learning (VGG19, ResNet50) sur IRM pour détecter les stades de démence.</p><a href="https://view.officeapps.live.com/op/view.aspx?src=https://raw.githubusercontent.com/Stagiaire2023GayeAlioune/Mon_application_AutoML/refs/heads/master/Detection_Alzheimer_Deep_Learning.docx">Rapport</a></div>""", unsafe_allow_html=True)
with col2:
    st.image("cancer.PNG", caption="Cancer du Sein", use_container_width=True)
    st.markdown("""<div class="project-box"><h3>Détection du Cancer du Sein</h3><p>Classification échographique des masses mammaires (bénin, malin, normal).</p><a href="https://github.com/Stagiaire2023GayeAlioune/Mon_application_AutoML/blob/master/Rapport_Cancer_du_sein.pdf">Rapport</a></div>""", unsafe_allow_html=True)
with col3:
    st.image("carte.PNG", caption="Fraude Bancaire", use_container_width=True)
    st.markdown("""<div class="project-box"><h3>Détection de Fraude Bancaire</h3><p>Classification des transactions frauduleuses via modèles supervisés.</p><a href="https://github.com/Stagiaire2023GayeAlioune/Mon_application_AutoML/blob/master/Rapport_detection_fraude.pdf">Rapport</a></div>""", unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    st.image("credi.jpg", caption="Analyse de Crédit", use_container_width=True)
    st.markdown("""<div class="project-box"><h3>Analyse des Risques de Crédit</h3><p>Scoring et prévision de risque client par ML.</p><a href="https://risquedecreditsclients.streamlit.app/">Application</a></div>""", unsafe_allow_html=True)
with col2:
    st.image("RH.PNG", caption="Dashboard RH", use_container_width=True)
    st.markdown("""<div class="project-box"><h3>Tableau de Bord RH</h3><p>Dashboard interactif pour analyser attrition, performance et démographie RH.</p><a href="https://applicationtableaudebordanalyserh.streamlit.app/">Application</a></div>""", unsafe_allow_html=True)

st.markdown("""
<div class="project-box"><h3>CRM Synergie Marketing Group</h3><p>CRM complet pour la gestion clients, ventes et commissions. Stack : Node.js, React, PostgreSQL, WebSocket.</p></div>
<div class="project-box"><h3>API OCR & LLM Immobilier</h3><p>Extraction automatique de données structurées à partir de PDF et images grâce à l’OCR et aux LLM.</p></div>
<div class="project-box"><h3>Agent IA Juridique Multilingue</h3><p>Assistant IA bilingue basé sur un pipeline RAG (OpenAI + FAISS) pour répondre à des questions juridiques.</p></div>
""", unsafe_allow_html=True)

# -----------------------------------------------------
# CONTACT
# -----------------------------------------------------
st.markdown("<h2 id='contact' style='color:#f25287;'>📞 Contact</h2>", unsafe_allow_html=True)
st.markdown("""
**📧 Email :** aliounegaye911@gmail.com  
**📱 Téléphone :** +33 7 63 55 69 82  
**🔗 LinkedIn :** [Alioune Gaye](https://www.linkedin.com/in/alioune-gaye-1a5161172/)
""")

# -----------------------------------------------------
# PIED DE PAGE
# -----------------------------------------------------
st.markdown("""
<footer>
© 2025 Data Workers — <b>Alioune Gaye</b> | Tous droits réservés.
</footer>
""", unsafe_allow_html=True)
