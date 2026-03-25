import streamlit as st

# -----------------------------------------------------
# CONFIGURATION GLOBALE
# -----------------------------------------------------
st.set_page_config(page_title="Data Workers", layout="wide")

# Effet de transition globale entre les pages
st.markdown("""
    <style>
    .main {
        opacity: 0;
        animation: fadeInAnimation ease 1.2s;
        animation-fill-mode: forwards;
    }

    @keyframes fadeInAnimation {
        0% { opacity: 0; transform: translateY(10px); }
        100% { opacity: 1; transform: translateY(0); }
    }

    /* Hover stylé pour les services et projets */
    .service-box:hover, .project-box:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 15px rgba(242, 82, 135, 0.25);
        transition: all 0.3s ease-in-out;
    }
    </style>

    <script>
    const observer = new MutationObserver(() => {
        const main = document.querySelector('.main');
        if (main) {
            main.style.opacity = '0';
            main.style.animation = 'none';
            void main.offsetWidth;
            main.style.animation = 'fadeInAnimation ease 1.2s';
            main.style.animationFillMode = 'forwards';
        }
    });
    observer.observe(document.body, { childList: true, subtree: true });
    </script>
""", unsafe_allow_html=True)

# Navigation latérale
page = st.sidebar.radio("Navigation", ["Les services que je propose", "À propos de moi", "Mes projets"])

# -----------------------------------------------------
# PAGE 1 : LES SERVICES QUE JE PROPOSE
# -----------------------------------------------------
if page == "Les services que je propose":
    # --- Bannière Hero ---
    st.markdown("""
        <style>
        .hero {
            background: linear-gradient(135deg, #f25287 0%, #ff8ba7 50%, #ffd1dc 100%);
            color: white;
            padding: 60px 20px;
            border-radius: 15px;
            text-align: center;
            box-shadow: 0 4px 10px rgba(0,0,0,0.2);
            margin-bottom: 40px;
            animation: fadeIn 1.5s ease-in-out;
        }
        .hero img {
            width: 120px;
            height: 120px;
            border-radius: 50%;
            object-fit: cover;
            margin-bottom: 15px;
            border: 3px solid white;
        }
        .hero h1 {
            font-size: 36px;
            font-weight: 800;
            margin-bottom: 10px;
        }
        .hero h2 {
            font-size: 20px;
            font-weight: 500;
            color: #fff;
            margin-top: 0;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(-15px); }
            to { opacity: 1; transform: translateY(0); }
        }
        </style>

        <div class="hero">
            <img src="https://raw.githubusercontent.com/Stagiaire2023GayeAlioune/Mon_application_AutoML/master/dv_lottery.jpg" alt="Alioune Gaye">
            <h1>Alioune Gaye</h1>
            <h2>Data Scientist | Consultant | Full-Stack AI Developer</h2>
        </div>
    """, unsafe_allow_html=True)

    # --- Animation machine à écrire cyclique ---
    st.markdown("""
        <style>
        .typewriter-container {
            width: 100%;
            text-align: center;
            font-size: 22px;
            font-weight: 600;
            color: #f25287;
            margin-top: -20px;
            height: 35px;
        }
        .typewriter-text {
            display: inline-block;
            border-right: 3px solid #f25287;
            white-space: nowrap;
            overflow: hidden;
            animation: typing 3s steps(40, end), blink .8s step-end infinite;
        }
        @keyframes typing { from { width: 0; } to { width: 100%; } }
        @keyframes blink { 50% { border-color: transparent; } }
        </style>

        <div class="typewriter-container">
            <span id="typewriter" class="typewriter-text"></span>
        </div>

        <script>
        const texts = [
            "Data Science 💡",
            "Développement Web 🌐",
            "Intelligence Artificielle 🤖",
            "Automatisation & Analyse de données 📊"
        ];
        let index = 0, charIndex = 0, currentText = "", isDeleting = false;
        const element = document.getElementById("typewriter");
        function type() {
            const fullText = texts[index];
            currentText = isDeleting ? fullText.substring(0, charIndex--) : fullText.substring(0, charIndex++);
            element.textContent = currentText;
            if (!isDeleting && charIndex === fullText.length) setTimeout(() => isDeleting = true, 1000);
            else if (isDeleting && charIndex === 0) { isDeleting = false; index = (index + 1) % texts.length; }
            setTimeout(type, isDeleting ? 60 : 120);
        }
        window.addEventListener('load', type);
        </script>
    """, unsafe_allow_html=True)

    # --- Présentation ---
    st.markdown("""
    <div style="text-align:center; margin-top:25px;">
    Je conçois et déploie des **solutions data et web intelligentes** alliant **analyse de données**, **intelligence artificielle**, 
    **automatisation** et **développement full-stack** pour accompagner la transformation numérique des entreprises.
    </div>
    """, unsafe_allow_html=True)

    # --- Services ---
    st.markdown("### 🌟 Mes domaines d’expertise")

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

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""<div class="service-box"><div class="emoji">📊</div><h3>Tableaux de bord</h3><p>Dashboards interactifs avec Power BI, Streamlit, Shiny et React pour piloter vos KPIs.</p></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""<div class="service-box"><div class="emoji">🗂️</div><h3>Gestion de données</h3><p>Architecture SQL/NoSQL, pipelines d’ingestion et optimisation de requêtes.</p></div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""<div class="service-box"><div class="emoji">🎓</div><h3>Formation</h3><p>Formations sur mesure en Data Science, IA, automatisation et développement full-stack.</p></div>""", unsafe_allow_html=True)

    # Réalisations récentes
    st.markdown("### 🚀 Réalisations récentes")
    st.markdown("""
    - **CRM Synergie Marketing Group** : système complet de gestion clients et ventes (Node.js, React, PostgreSQL, WebSocket).  
    - **API OCR & LLM Immobilier** : extraction et validation automatique de documents administratifs (CNI, bulletins, contrats).  
    - **Agent IA Juridique Multilingue** : assistant intelligent basé sur un pipeline RAG (OpenAI + FAISS).  
    """)

# -----------------------------------------------------
# PAGE 2 : À PROPOS DE MOI
# -----------------------------------------------------
elif page == "À propos de moi":
    st.markdown("<h1 style='text-align:center;'>À propos de moi</h1>", unsafe_allow_html=True)
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image("dv_lottery.jpg", use_container_width=True)
    with col2:
        st.markdown("""
        Je suis **Alioune Gaye**, Data Scientist, statisticien et développeur full-stack spécialisé en IA et automatisation.  
        Mon objectif est de transformer la donnée en valeur métier à travers des solutions concrètes, performantes et évolutives.
        """)
    st.markdown("### 🎓 Éducation")
    st.markdown("**Master en Statistique, Modélisation et Science des données** – Université Claude Bernard Lyon 1 (Bac +5).")

    st.markdown("### 💡 Compétences comportementales")
    st.markdown("""
    - Communication claire, esprit d’équipe et autonomie  
    - Rigueur, innovation et adaptabilité  
    - Leadership technique et accompagnement pédagogique
    """)

    st.markdown("### 🧠 Compétences techniques")
    st.markdown("""
    - **IA & Data Science :** Machine Learning, Deep Learning, NLP, Vision, Séries temporelles  
    - **Backend :** Node.js, Express, TypeScript  
    - **Frontend :** React, Vite, Tailwind CSS, shadcn, Radix UI  
    - **Base de données :** PostgreSQL, Drizzle ORM, MySQL, SQL Server  
    - **Outils :** Docker, Git, Azure, Django, Scrapy  
    - **Automatisation :** cron jobs, scripts Python, ETL, APIs REST  
    - **Docs & PDF :** jspdf, html2pdf, docxtemplater, mammoth  
    """)

    st.markdown("### 💻 Langages")
    st.markdown("Python, R, C++, SQL, TypeScript, JavaScript, Stata")

    st.markdown("### 📊 Outils de visualisation")
    st.markdown("Power BI, Excel, Streamlit, Shiny, Tableau")

    st.markdown("### ☁️ Cloud & Collaboration")
    st.markdown("Azure, GitHub, Google Cloud, SendGrid, AWS S3")

# -----------------------------------------------------
# PAGE 3 : MES PROJETS
# -----------------------------------------------------
elif page == "Mes projets":
    st.markdown("<h1 style='text-align:center;'>Mes Projets</h1>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image("Alzeimer.PNG", caption="Détection Alzheimer", use_container_width=True)
        st.markdown("""
        <div class="project-box"><h3>Détection de la Maladie d'Alzheimer</h3>
        <p>Deep Learning (VGG19, ResNet50) sur IRM pour détecter les stades de démence.</p>
        <a href="https://view.officeapps.live.com/op/view.aspx?src=https://raw.githubusercontent.com/Stagiaire2023GayeAlioune/Mon_application_AutoML/refs/heads/master/Detection_Alzheimer_Deep_Learning.docx">Rapport</a></div>
        """, unsafe_allow_html=True)
    with col2:
        st.image("cancer.PNG", caption="Cancer du Sein", use_container_width=True)
        st.markdown("""
        <div class="project-box"><h3>Détection du Cancer du Sein</h3>
        <p>Classification échographique des masses mammaires (bénin, malin, normal).</p>
        <a href="https://github.com/Stagiaire2023GayeAlioune/Mon_application_AutoML/blob/master/Rapport_Cancer_du_sein.pdf">Rapport</a></div>
        """, unsafe_allow_html=True)
    with col3:
        st.image("carte.PNG", caption="Fraude Bancaire", use_container_width=True)
        st.markdown("""
        <div class="project-box"><h3>Détection de Fraude Bancaire</h3>
        <p>Classification des transactions frauduleuses via modèles supervisés.</p>
        <a href="https://github.com/Stagiaire2023GayeAlioune/Mon_application_AutoML/blob/master/Rapport_detection_fraude.pdf">Rapport</a></div>
        """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.image("credi.jpg", caption="Analyse des Risques de Crédit", use_container_width=True)
        st.markdown("""
        <div class="project-box"><h3>Analyse des Risques de Crédit</h3>
        <p>Scoring de solvabilité et prévision du risque client par ML.</p>
        <a href="https://risquedecreditsclients.streamlit.app/">Application</a></div>
        """, unsafe_allow_html=True)
    with col2:
        st.image("RH.PNG", caption="Dashboard RH", use_container_width=True)
        st.markdown("""
        <div class="project-box"><h3>Tableau de Bord RH</h3>
        <p>Dashboard interactif pour analyser attrition, performance et démographie RH.</p>
        <a href="https://applicationtableaudebordanalyserh.streamlit.app/">Application</a></div>
        """, unsafe_allow_html=True)

    st.markdown("### 🔬 Projets avancés")
    col1, col2 = st.columns(2)
    with col1:
        st.image("crm_synergie.png", caption="CRM Synergie Marketing Group", use_container_width=True)
        st.markdown("""
        <div class="project-box"><h3>CRM Synergie Marketing Group</h3>
        <p>CRM complet pour la gestion clients, ventes et commissions. Stack : Node.js, React, PostgreSQL, WebSocket.</p></div>
        """, unsafe_allow_html=True)
    with col2:
        st.image("api_ocr.png", caption="API OCR & LLM", use_container_width=True)
        st.markdown("""
        <div class="project-box"><h3>API OCR & LLM pour documents immobiliers</h3>
        <p>Extraction automatique de données structurées à partir de PDF et images grâce à l’OCR et aux LLM.</p></div>
        """, unsafe_allow_html=True)

    st.image("ai_juridique.png", caption="Agent IA Juridique", use_container_width=True)
    st.markdown("""
    <div class="project-box"><h3>Agent IA Juridique Multilingue</h3>
    <p>Assistant IA bilingue (français/arabe) basé sur un pipeline RAG, embeddings FAISS et OpenAI pour répondre à des questions juridiques à partir de documents internes.</p></div>
    """, unsafe_allow_html=True)

# -----------------------------------------------------
# PIED DE PAGE
# -----------------------------------------------------
st.markdown("---")
st.markdown("""
<p style='text-align: center;'>
    <strong>Mes contacts :</strong><br>
    <a href='https://www.linkedin.com/in/alioune-gaye-1a5161172/' target='_blank' style='margin-right: 15px;'>
        <img src='https://upload.wikimedia.org/wikipedia/commons/c/ca/LinkedIn_logo_initials.png' style='width:20px;'> LinkedIn
    </a>
    <a href='tel:+33763556982' style='margin-right: 15px;'>
        <img src='https://upload.wikimedia.org/wikipedia/commons/6/6c/Phone_icon.png' style='width:20px;'> 0763556982
    </a>
    <a href='mailto:aliounegaye911@gmail.com'>
        <img src='https://upload.wikimedia.org/wikipedia/commons/2/27/Android_Email_4.4_Icon.png' style='width:20px;'> aliounegaye911@gmail.com
    </a><br><br>
    © 2025 Data Workers – <strong>Alioune Gaye</strong>.
</p>
""", unsafe_allow_html=True)
