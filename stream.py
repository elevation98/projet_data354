import streamlit as st
import os
import json
from chatbot import chatbot_faiss  # Fonction du chatbot
from googlesearch import search  # Fonction pour la recherche sur le web
from streamlit.runtime.scriptrunner import RerunException

# Initialisation de l'état de session
if "question" not in st.session_state:
    st.session_state["question"] = ""

# Configuration de la page
st.set_page_config(page_title="Chatbot IA", layout="wide")
st.sidebar.title("⚙️ Options")

# Ajout d'une fonctionnalité de chargement de fichier
doc_file = st.sidebar.file_uploader("📂 Charger un fichier", type=["txt", "pdf", "csv"])
if doc_file:
    st.sidebar.success("Fichier chargé avec succès!")  # Ajoutez un traitement des fichiers ici

# Ajout d'un bouton pour recharger la page
if st.sidebar.button("🔄 Recharger la page"):
    raise RerunException("Page rechargée.")  # Forcer le rechargement de la page

# Affichage du titre principal
st.title("🤖 Chatbot IA - Recherche de documents")

# Champ de saisie pour la question (sans conflit de clé)
question = st.text_area(
    "Posez votre question ici :", 
    value=st.session_state["question"],  # Utilise la valeur actuelle
    height=150,
    key="input_question"  # Clé différente pour éviter les conflits
)

# Option pour la recherche web
use_web_search = st.sidebar.checkbox("🔍 Rechercher sur le web si besoin")

# Bouton de recherche
if st.button("Rechercher"):
    if question:  # Vérifie si une question est saisie
        response = chatbot_faiss(question)  # Réponse du chatbot
        if use_web_search and not response:
            response = search(question)  # Recherche web si aucune réponse du bot

        # Affichage de la réponse
        st.markdown(f"**Bot:** {response}")

        # Réinitialisation du champ de texte
        st.session_state["question"] = ""
    else:
        st.warning("Veuillez entrer une question.")

# Fonction pour effacer l'historique
def clear_history():
    st.session_state.clear()  # Réinitialise tout l'état
    raise RerunException("Application refreshed.")

# Bouton pour effacer l'historique
if st.sidebar.button("🗑 Effacer l'historique"):
    clear_history()

