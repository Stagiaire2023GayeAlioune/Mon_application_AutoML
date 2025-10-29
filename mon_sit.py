import streamlit as st
from streamlit_lottie import st_lottie
import requests

# -----------------------------------------------------
# CONFIGURATION GLOBALE
# -----------------------------------------------------
st.set_page_config(page_title="Data Workers", layout="wide", page_icon="💚")

# -----------------------------------------------------
# STYLE GLOBAL + TRANSITIONS
# -----------------------------------------------------
st.markdown("""
    <style>
    html, body {font-family: 'Poppins', sans-serif; background-color: #fafafa;}

    /* Barre de navigation sticky */
    .navbar {
        position: fixed;
        top: 0; left: 0; width: 100%;
        background-color: white;
        padding: 12px 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        z-index: 100;
        text-align: center;
    }
    .navbar a {
        margin: 0 20px;
        text-decoration: none;
        font-weight: 600;
        color: #00c39a;
        transition: color 0.3s;
    }
    .navbar a:hover { color: #007e6c; }

    /* Contenu principal avec animation d’apparition */
    .main {
        opacity: 0;
        animation: fadeIn 1.2s ease forwards;
    }
    @keyframes fadeIn {
        0% {opacity: 0; transform: translateY(10px);}
        100% {opacity: 1; transform: translateY(0);}
    }

    /* HERO banner */
    .hero {
        background: linear-gradient(120deg, #00c39a, #00d6a3, #00b38a);
        background-size: 300% 300%;
        animation: gradientShift 8s ease infinite;
        padding: 80px 20px;
        color: white;
        text-align: center;
        border-radius: 20px;
        margin-top: 80px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.15);
    }
    @keyframes gradientShift {
        0% {background-position: 0% 50%;}
        50% {background-position: 100% 50%;}
        100% {background-position: 0% 50%;}
    }

    .hero h1 { font-size: 40px; font-weight: 800; margin-bottom: 10px; }
    .hero h2 { font-size: 22px; font-weight: 500; margin-top: 0; }
    .hero a {
        background-color: white;
        color: #00c39a;
        padding: 12px 24px;
        border-radius: 8px;
        text-decoration: none;
        font-weight: 600;
        display: inline-block;
        margin-top: 20px;
        transition: all 0.3s ease;
    }
    .hero a:hover { background-color: #007e6c; color: white; }

    /* Cartes de services et projets */
    .service-box, .project-box {
        border: 1px solid #ddd;
        border-radius: 12px;
        padding: 20px;
        margin: 10px;
        background-color: white;
        box-shadow: 0 2px 6px rgba(0,0,0,0.08);
        transition: all 0.3s ease;
        text-align: center;
    }
    .service-box:hover, .project-box:hover {
        transform: translateY(-6px);
        box-shadow: 0 8px 20px rgba(0,195,154,0.25);
    }
    .service-box h3, .project-box h3 {
        color: #333; font-weight: 700; margin-bottom: 10px;
    }
    .service-box p, .project-box p {
        color: #666; font-size: 14px;
    }
    .emoji {font-size: 38px; margin-bottom: 10px;}
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------
# NAVBAR
# -----------------------------------------------------
st.markdown("""
<div class="navbar">
    <a href="#services">Services</a>
    <a href="#apropos">À propos</a>
    <a href="#projets">Projets</a>
    <a href="#contact">Contact</a>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------
# HERO SECTION
# -----------------------------------------------------
st.markdown("""
<div class="hero">
    <h1>Alioune Gaye</h1>
    <h2>L’intelligence artificielle et la donnée au cœur de votre réussite.</h2>
    <a href="mailto:aliounegaye911@gmail.com">💌 Me contacter</a>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------
# LOTTIE ANIMATIONS
# -----------------------------------------------------
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

ai_anim = load_lottieurl("https://lottie.host/da1b6f23-29b4-4c0c-80af-1c0a4b82b8e3/Z4aQxNGRxA.json")
data_anim = load_lottieurl("https://lottie.host/f522a5b1-7f6b-4236-97c2-4e99cb469ea8/1qGtbUPpUl.json")
web_anim = load_lottieurl("https://lottie.host/2e896d9f-bb7d-4c7a-b7de-58d7b42d8e45/yo6nbpQJ9M.json")

st.markdown("<div id='services'></div>", unsafe_allow_html=True)
st.subheader("🌟 Mes domaines d’expertise")

col1, col2, col3 = st.columns(3)
with col1:
    st_lottie(ai_anim, height=150, key="ai")
    st.markdown("""<div class="service-box"><div class="emoji">🤖</div><h3>IA & Modélisation</h3><p>Conception de modèles prédictifs et automatisation intelligente via Machine Learning et Deep Learning.</p></div>""", unsafe_allow_html=True)
with col2:
    st_lottie(data_anim, height=150, key="data")
    st.markdown("""<div class="service-box"><div class="emoji">📊</div><h3>Analyse de données</h3><p>Nettoyage, structuration et visualisation des données pour révéler les insights les plus pertinents.</p></div>""", unsafe_allow_html=True)
with col3:
    st_lottie(web_anim, height=150, key="web")
    st.markdown("""<div class="service-box"><div class="emoji">💻</div><h3>Développement Web</h3><p>Création d’applications web et d’API avec Node.js, React, Tailwind CSS et PostgreSQL.</p></div>""", unsafe_allow_html=True)

# -----------------------------------------------------
# À PROPOS DE MOI
# -----------------------------------------------------
st.markdown("<div id='apropos'></div>", unsafe_allow_html=True)
st.subheader("👤 À propos de moi")

col1, col2 = st.columns([1, 2])
with col1:
    st.image("dv_lottery.jpg", caption="Alioune Gaye", use_container_width=True)
with col2:
    st.markdown("""
    Je suis **Alioune Gaye**, Data Scientist et Développeur Full-Stack passionné par l’intelligence artificielle et la valorisation de la donnée.  
    Mon objectif : transformer des données brutes en leviers stratégiques pour les entreprises à travers des solutions IA innovantes.
    """)

st.markdown("### 🎓 Formation")
st.markdown("**Master en Statistique, Modélisation et Science des données** – Université Claude Bernard Lyon 1 (Bac +5)")

st.markdown("### 🧠 Compétences clés")
st.markdown("""
- **Backend :** Node.js, Express, TypeScript, PostgreSQL, Drizzle ORM  
- **Frontend :** React, Tailwind CSS, shadcn, Radix UI  
- **IA :** Machine Learning, Deep Learning, NLP, Vision, OCR, LLMs (RAG)  
- **Outils :** Docker, Azure, GitHub, SendGrid, AWS S3  
- **Automatisation :** cron jobs, API REST, ETL  
- **Docs & PDF :** jspdf, docxtemplater, mammoth, html2pdf  
""")

# -----------------------------------------------------
# MES PROJETS
# -----------------------------------------------------
st.markdown("<div id='projets'></div>", unsafe_allow_html=True)
st.subheader("🚀 Mes projets récents")

col1, col2, col3 = st.columns(3)
with col1:
    st.image("Alzeimer.PNG", caption="Détection Alzheimer", use_container_width=True)
    st.markdown("""<div class="project-box"><h3>Détection Alzheimer</h3><p>Deep Learning (VGG19, ResNet50) sur IRM pour détecter les stades de démence.</p></div>""", unsafe_allow_html=True)
with col2:
    st.image("crm_synergie.png", caption="CRM Synergie Marketing", use_container_width=True)
    st.markdown("""<div class="project-box"><h3>CRM Synergie Marketing Group</h3><p>CRM complet de gestion clients, ventes et commissions – Stack : Node.js, React, PostgreSQL, WebSocket.</p></div>""", unsafe_allow_html=True)
with col3:
    st.image("api_ocr.png", caption="API OCR & LLM", use_container_width=True)
    st.markdown("""<div class="project-box"><h3>API OCR & LLM Immobilier</h3><p>Extraction automatique de documents (PDF, images) via OCR + modèles LLM pour vérification.</p></div>""", unsafe_allow_html=True)

st.image("ai_juridique.png", caption="Agent IA Juridique", use_container_width=True)
st.markdown("""<div class="project-box"><h3>Agent IA Juridique</h3><p>Assistant IA bilingue (français/arabe) basé sur RAG et embeddings OpenAI pour répondre à des questions juridiques.</p></div>""", unsafe_allow_html=True)

# -----------------------------------------------------
# CONTACT / FOOTER
# -----------------------------------------------------
st.markdown("<div id='contact'></div>", unsafe_allow_html=True)
st.subheader("📬 Me contacter")

st.markdown("""
<p style='text-align: center; font-size: 18px;'>
    📧 <a href='mailto:aliounegaye911@gmail.com'>aliounegaye911@gmail.com</a><br>
    📞 <a href='tel:+33763556982'>07 63 55 69 82</a><br>
    🌐 <a href='https://www.linkedin.com/in/alioune-gaye-1a5161172/'>LinkedIn</a><br><br>
    © 2025 Data Workers – <strong>Alioune Gaye</strong>
</p>
""", unsafe_allow_html=True)
