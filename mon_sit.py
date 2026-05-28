import streamlit as st

# -----------------------------------------------------
# CONFIGURATION GLOBALE
# -----------------------------------------------------
st.set_page_config(page_title="Data Workers", layout="wide")

# Effet de transition globale entre les pages
st.markdown("""
    <style>
    :root {
        --primary: #ec407a;
        --primary-dark: #c2185b;
        --accent: #7c4dff;
        --bg-soft: #f7f9fc;
        --text-main: #1f2937;
        --text-muted: #6b7280;
        --card-border: #e5e7eb;
    }

    .stApp {
        background: linear-gradient(180deg, #ffffff 0%, var(--bg-soft) 100%);
        color: var(--text-main);
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1250px;
    }

    .main {
        opacity: 0;
        animation: fadeInAnimation ease 1.2s;
        animation-fill-mode: forwards;
    }

    @keyframes fadeInAnimation {
        0% { opacity: 0; transform: translateY(10px); }
        100% { opacity: 1; transform: translateY(0); }
    }

    /* Cartes services/projets */
    .service-box, .project-box {
        background: #ffffff;
        border: 1px solid var(--card-border);
        border-radius: 16px;
        padding: 18px 18px 14px 18px;
        margin-bottom: 14px;
        box-shadow: 0 4px 14px rgba(15, 23, 42, 0.06);
        min-height: 170px;
    }

    .service-box h3, .project-box h3 {
        margin-top: 4px;
        margin-bottom: 10px;
        color: #111827;
        font-size: 1.1rem;
    }

    .service-box p, .project-box p {
        color: var(--text-muted);
        line-height: 1.5;
        margin-bottom: 9px;
        font-size: 0.96rem;
    }

    .emoji {
        font-size: 1.45rem;
        margin-bottom: 8px;
    }

    .project-box a {
        color: var(--primary-dark);
        text-decoration: none;
        font-weight: 600;
    }

    .project-box a:hover {
        text-decoration: underline;
    }

    /* Hover stylé pour les services et projets */
    .service-box:hover, .project-box:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 24px rgba(124, 77, 255, 0.16);
        border-color: #d8ccff;
        transition: all 0.25s ease-in-out;
    }

    /* Images plus professionnelles */
    [data-testid="stImage"] img {
        border-radius: 14px;
        border: 1px solid #e8eaf0;
        box-shadow: 0 5px 16px rgba(15, 23, 42, 0.10);
    }

    /* Titres */
    h1, h2, h3 {
        letter-spacing: -0.2px;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #141a2e 0%, #1f2940 100%);
    }

    section[data-testid="stSidebar"] * {
        color: #f3f4f6 !important;
    }

    section[data-testid="stSidebar"] .stRadio > label {
        font-weight: 600;
        font-size: 0.95rem;
    }

    section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.08);
        padding: 8px 10px;
        border-radius: 10px;
        margin-bottom: 6px;
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
st.sidebar.markdown("---")
st.sidebar.markdown("""
### Alioune Gaye
**Data Scientist & Full-Stack AI Developer**

Accompagnement des entreprises sur :
- IA appliquée
- Automatisation métier
- Développement web & data products
""")

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
        st.markdown("""<div class="service-box"><div class="emoji">📈</div><h3>Data Analytics & BI</h3><p>Nettoyage, modélisation et visualisation de données pour transformer les KPIs en décisions actionnables.</p></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""<div class="service-box"><div class="emoji">🤖</div><h3>IA Générative & Prédictive</h3><p>LLM, RAG, OCR et modèles ML/DL pour automatiser les tâches métier et améliorer la performance opérationnelle.</p></div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""<div class="service-box"><div class="emoji">⚙️</div><h3>API & Automatisation</h3><p>Conception d'API robustes et workflows automatisés (CRON, workers, intégrations tierces, notifications).</p></div>""", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""<div class="service-box"><div class="emoji">💻</div><h3>Développement Backend</h3><p>Node.js, Express, TypeScript, Django, PostgreSQL/SQL Server, authentification sécurisée et services temps réel.</p></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""<div class="service-box"><div class="emoji">🎨</div><h3>Frontend Web & Mobile</h3><p>React, Next.js, Vite, Tailwind, shadcn/ui, React Native Expo pour des expériences fluides sur web et mobile.</p></div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""<div class="service-box"><div class="emoji">🧩</div><h3>Intégrations Métier</h3><p>WhatsApp, OAuth, Google APIs, SendGrid/Nodemailer, paiement, PDF/CSV/FEC/SEPA et outils SaaS.</p></div>""", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""<div class="service-box"><div class="emoji">📊</div><h3>Dashboards & Pilotage</h3><p>Power BI, Streamlit et dashboards web sur mesure pour suivre ventes, productivité, recrutement et finance.</p></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""<div class="service-box"><div class="emoji">🗂️</div><h3>Architecture Data</h3><p>Conception de schémas, migrations, ETL, qualité des données et optimisation SQL pour des systèmes fiables.</p></div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""<div class="service-box"><div class="emoji">🎓</div><h3>Formation & Mentorat</h3><p>Accompagnement sur mesure en développement full-stack, data/IA, bonnes pratiques projet et montée en compétence équipe.</p></div>""", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""<div class="service-box"><div class="emoji">🏢</div><h3>Solutions CRM Métier</h3><p>Conception de CRM adaptés au terrain : prospects, ventes, commissions, recrutement, facturation et suivi opérationnel.</p></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""<div class="service-box"><div class="emoji">☁️</div><h3>Cloud, DevOps & Qualité</h3><p>CI/CD, conteneurisation, monitoring, sécurité, performance et fiabilisation des applications en production.</p></div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""<div class="service-box"><div class="emoji">🛠️</div><h3>Conseil & Cadrage Produit</h3><p>Audit fonctionnel/technique, priorisation roadmap, définition MVP et architecture pour livrer plus vite avec impact.</p></div>""", unsafe_allow_html=True)

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
        Je suis **Alioune Gaye**, Data Scientist, statisticien et développeur full-stack orienté produits IA, automatisation métier et plateformes web/mobiles.  
        J'interviens de bout en bout : cadrage du besoin, architecture technique, développement, déploiement et amélioration continue.
        
        Mon objectif est de transformer la donnée en valeur métier à travers des solutions concrètes, robustes, performantes et évolutives.
        """)
    st.markdown("### 🎓 Éducation")
    st.markdown("""
    - **Master en Statistique, Modélisation et Science des données** – Université Claude Bernard Lyon 1 (Bac +5)
    - **Formation Développeur Full-Stack** : conception d'applications web, API, bases de données, intégration et bonnes pratiques de production
    """)

    st.markdown("### 🧭 Domaines d'intervention")
    st.markdown("""
    - Développement de plateformes métier (CRM, marketing automation, outils de pilotage)
    - Industrialisation de workflows data/IA pour les équipes opérationnelles
    - Intégration d'outils multicanaux (WhatsApp, réseaux sociaux, email, API tierces)
    - Conception d'interfaces modernes orientées expérience utilisateur
    - Structuration des données, reporting, dashboards et KPIs décisionnels
    """)

    st.markdown("### 💡 Compétences comportementales")
    st.markdown("""
    - Communication claire, esprit d'équipe et collaboration transverse  
    - Rigueur, ownership, fiabilité en production  
    - Capacité de vulgarisation, pédagogie et accompagnement des utilisateurs  
    - Résolution de problèmes complexes et prise de décision orientée impact
    """)

    st.markdown("### 🧠 Compétences techniques")
    st.markdown("""
    - **IA & Data Science :** Machine Learning, Deep Learning, NLP, Vision, séries temporelles, RAG, embeddings (FAISS), OCR (Tesseract), OpenAI, Gemini  
    - **Backend :** Node.js, Express, TypeScript, Django, sessions/auth, middleware métier, APIs REST  
    - **Frontend Web :** React 18, Next.js, Vite, Tailwind CSS, shadcn/ui, Radix UI, TanStack Query  
    - **Mobile :** React Native, Expo (Android/iOS)  
    - **Temps réel :** WebSocket, notifications, traitements asynchrones, workers, Bull/BullMQ  
    - **Base de données :** PostgreSQL, SQL Server, MySQL, Drizzle ORM, modélisation de schémas, migrations  
    - **Messaging & intégrations :** WhatsApp Web, OAuth, Google APIs/Calendar, SendGrid, Nodemailer  
    - **Documents & exports :** PDF/CSV/FEC/SEPA, docxtemplater, html2pdf, génération de rapports automatisés  
    - **DevOps & qualité :** Git/GitHub, Docker, CI/CD, tests, logging, monitoring, optimisation des performances  
    - **Automatisation :** CRON métiers, scripts Python, pipelines ETL et orchestration de tâches
    """)

    st.markdown("### 💻 Langages")
    st.markdown("Python, TypeScript, JavaScript, SQL, R, C++, Stata")

    st.markdown("### 📊 Outils de visualisation")
    st.markdown("Power BI, Streamlit, Tableau, Shiny, Excel")

    st.markdown("### ☁️ Cloud & Collaboration")
    st.markdown("Azure, Google Cloud, AWS (S3), GitHub, Postman, Notion, Jira")

# -----------------------------------------------------
# PAGE 3 : MES PROJETS
# -----------------------------------------------------
elif page == "Mes projets":
    st.markdown("""
    <style>
    .projects-title {
        text-align: center;
        margin-bottom: 8px;
    }
    .projects-subtitle {
        text-align: center;
        color: #6b7280;
        margin-bottom: 28px;
    }
    .project-section-title {
        margin-top: 22px;
        margin-bottom: 12px;
    }
    /* Uniformiser les visuels uniquement sur la page projets */
    [data-testid="stImage"] img {
        width: 100%;
        height: 230px;
        object-fit: cover;
        border-radius: 14px;
    }
    [data-testid="stCaptionContainer"] {
        margin-bottom: 8px;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("<h1 class='projects-title'>Mes Projets</h1>", unsafe_allow_html=True)
    st.markdown("<p class='projects-subtitle'>Portfolio structuré : Data/IA, produits digitaux et plateformes métier.</p>", unsafe_allow_html=True)

    st.markdown("### 📊 Projets Data & IA", unsafe_allow_html=False)
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
    col1, col2, col3 = st.columns(3)
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
    with col3:
        st.image("ai_juridique.png", caption="Agent IA Juridique", use_container_width=True)
        st.markdown("""
        <div class="project-box"><h3>Agent IA Juridique Multilingue</h3>
        <p>Assistant IA bilingue (français/arabe) basé sur un pipeline RAG, embeddings FAISS et OpenAI pour répondre à des questions juridiques à partir de documents internes.</p></div>
        """, unsafe_allow_html=True)

    st.markdown("### 📱 Projets plateformes & automatisation")
    col1, col2 = st.columns(2)
    with col1:
        waaw_img1, waaw_img2 = st.columns(2)
        with waaw_img1:
            st.image("Capture d’écran 2026-05-28 094939.png", caption="WaaW - Connexion", use_container_width=True)
        with waaw_img2:
            st.image("PAGE d'acceuil.png", caption="WaaW - Accueil", use_container_width=True)
        st.markdown("""
        <div class="project-box"><h3>WaaW — Plateforme Social + WhatsApp</h3>
        <p><strong>Vision :</strong> WaaW est une plateforme web qui permet à une entreprise de créer, organiser et diffuser du contenu marketing multicanal depuis un seul espace.</p>
        <p><strong>Publication multi-réseaux :</strong> l'utilisateur prépare un message, ajoute des médias (image, vidéo, document, audio), sélectionne les canaux connectés puis lance la diffusion.</p>
        <p><strong>Connexion des canaux :</strong> module OAuth pour lier WhatsApp, Telegram, Slack et d'autres réseaux, ensuite exploitables dans les campagnes.</p>
        <p><strong>Focus WhatsApp :</strong> connexion WhatsApp Web par QR code, récupération des contacts/groupes, diffusion de messages promotionnels en masse et envoi avec pièces jointes.</p>
        <p><strong>Pilotage :</strong> historique complet (filtrer, consulter, modifier, relancer un envoi échoué, supprimer/restaurer) et statistiques par réseau avec insights analytiques + export PDF.</p>
        <p><strong>Création vidéo marketing :</strong> workflow guidé pour saisir l'offre, charger les médias, générer la vidéo, suivre la progression et publier le résultat final.</p>
        <p><strong>Positionnement :</strong> outil d'automatisation marketing et de gestion de diffusion multicanale, avec un fort accent sur WhatsApp et la performance des campagnes.</p>
        <p><a href="https://www.waaw.cloud/" target="_blank">Accéder à l'application WaaW</a></p></div>
        """, unsafe_allow_html=True)
    with col2:
        st.image("veta.png", caption="Vetafrik - Page d'accueil", use_container_width=True)
        st.markdown("""
        <div class="project-box"><h3>Vetafrik — Site vitrine & serveur WhatsApp</h3>
        <p><strong>Vision :</strong> Vetafrik est un projet orienté vitrine commerciale et prise de commande simple dans la nutrition animale.</p>
        <p><strong>Objectif :</strong> présenter les produits, informer les éleveurs, générer des demandes commerciales et faciliter la commande via WhatsApp.</p>
        <p><strong>Parcours utilisateur :</strong> découverte de la marque, consultation des produits par catégorie/espèce, lecture des conseils, ajout au panier puis validation de commande.</p>
        <p><strong>Processus commande :</strong> après validation, la commande est transmise via un flux opérationnel WhatsApp.</p>
        <p><strong>Fonction clé :</strong> génération automatique d'un bon de commande PDF (articles, quantités, total, devise), envoyé au client sur WhatsApp avec copie à l'équipe Vetafrik.</p>
        <p><strong>Développement commercial :</strong> formulaires de contact et <em>devenir distributeur</em> pour soutenir la relation client et l'extension du réseau.</p>
        <p><strong>Public cible :</strong> éleveurs, distributeurs potentiels et acteurs agro-élevage en Afrique de l'Ouest (notamment Sénégal et Côte d'Ivoire).</p>
        <p><strong>Bénéfices métier :</strong> meilleure visibilité des produits, conversion simplifiée, standardisation des commandes (PDF + WhatsApp) et mise en relation plus fluide.</p>
        <p><strong>Stack technique :</strong> Next.js/React/TypeScript/Tailwind (FR-EN), SQL Server pour les formulaires et serveur Node.js pour WhatsApp + génération PDF.</p>
        <p><a href="https://vetafrik.com/fr" target="_blank">Accéder au site Vetafrik</a></p></div>
        """, unsafe_allow_html=True)

    st.markdown("### 🧾 Plateforme CRM métier")
    crm_col1, crm_col2 = st.columns([1, 1.3])
    with crm_col1:
        st.image("crm.png", caption="CRM - Page d'accueil", use_container_width=True)
    with crm_col2:
        st.markdown("""
        <div class="project-box"><h3>CRM Métier Full-Stack — Vente Terrain / Télécom (Free)</h3>
        <p><strong>Utilité :</strong> centraliser toute l'activité commerciale dans un seul outil, couvrir le cycle complet client (prospect -> vente -> installation -> facturation) et automatiser les commissions (CVD, MLM, CAE, CPC) avec traçabilité/audit.</p>
        <p><strong>Application concrète :</strong> gestion clients/prospects, détection des doublons, suivi des tâches, pilotage vendeurs/performances/projections et tunnel de recrutement multi-étapes (inscription, formation, attestation, contrat, finalisation).</p>
        <p><strong>Opérations financières :</strong> facturation/comptabilité (TVA, livres, rapprochement), verrouillage des factures, exports PDF/CSV/FEC/SEPA et conformité fiscale.</p>
        <p><strong>Collaboration :</strong> messagerie interne, passation de dossiers, bibliothèque de documents partagés, notifications automatiques et CRON métiers.</p>
        <p><strong>Public cible :</strong> administrateurs, commerciaux, recruteurs/managers MLM et clients finaux (module parrainage avec espace sécurisé).</p>
        <p><strong>Stack frontend :</strong> React 18 + TypeScript, Vite, Wouter, TailwindCSS, shadcn/ui (Radix), TanStack Query, React Hook Form + Zod.</p>
        <p><strong>Stack backend :</strong> Node.js, Express, TypeScript, sessions (<code>express-session</code>) et middleware métier.</p>
        <p><strong>Données & sécurité :</strong> PostgreSQL + Drizzle ORM/Kit, authentification session, rôles/permissions, cookies sécurisés, reset password.</p>
        <p><strong>Temps réel & intégrations :</strong> WebSocket, Daily (visioconf), Google APIs/Calendar, email (Nodemailer/SendGrid), OCR/IA (Gemini, OpenAI, Tesseract), génération PDF/CSV/SEPA.</p>
        <p><strong>Fonctionnalités majeures :</strong> CRM avancé, gestion stock/cartes SIM, commissions multi-systèmes, facturation admin, comptabilité assistée IA, recrutement complet, parrainage vendeur/client et analytics comportementales.</p></div>
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
