import json
import pandas as pd
import re
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
import spacy
import google.generativeai as gg
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from urllib.parse import urlparse
from langchain.chains import LLMChain


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

#Ce code configure:
#- une clé API pour accéder aux services Google,
#- utilise le modèle NLP sentence-transformers/all-MiniLM-L6-v2 pour la recherche sémantique, et
#- initialise le modèle génératif 'Gemini 1.5 Flash' pour des tâches comme la génération de texte et le résumé, formant ainsi une base puissante pour des applications d'IA.

GOOGLE_API_KEY = "AIzaSyAbSwlPr-rJkMHM7p2yu30yg1331dvQsc8"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
model = gg.GenerativeModel("gemini-1.5-flash")


# "gg.configure(api_key=GOOGLE_API_KEY)":    Elle est utilisée pour autoriser et authentifier les requêtes envoyées
#  à l'API Google via la bibliothèque gg. La clé API permet à  votre application de s'identifier auprès des services
#  de Google et d'utiliser les fonctionnalités associées (comme l'accès à des modèles ou à des outils spécifiques).

gg.configure(api_key=GOOGLE_API_KEY)


#-----------------------------------------------#

# Charger les données JSON avec encodage UTF-8
with open("articles_2.json", "r", encoding="utf-8") as f:
    data = json.load(f)
# Conversion de la liste en DataFrame
df = pd.DataFrame(data)

#------------------------------------------#

def preprocess_data(df):
    """
    Prétraite un DataFrame en effectuant les étapes suivantes :
    1. Supprime les lignes avec des valeurs manquantes.
    2. Supprime les doublons.
    3. Extrait des mots-clés de l'URL.
    4. Convertit les textes en minuscules.
    5. Supprime les stopwords.
    6. Lemmatise les mots.
    7. Normalise les données (emails, nombres, caractères spéciaux, etc.).
    8. Supprime les traits d'union entre les mots.

    Args:
        df (pd.DataFrame): Le DataFrame brut à prétraiter.

    Returns:
        pd.DataFrame: Le DataFrame prétraité.
    """
    # Supprimer les lignes avec valeurs manquantes
    df = df.dropna()

    # Supprimer les doublons
    df = df.drop_duplicates()

    # Extraire des mots-clés de l'URL
    df["mots_cles_lien"] = df["lien"].apply(lambda x: "  ".join(urlparse(str(x)).path.split("/")))

    # Supprimer les colonnes inutiles
    df = df.drop(columns=['auteur', 'date', 'lien'])

    # Conversion en minuscules
    df['titre'] = df['titre'].str.lower()
    df['texte_previsualisation'] = df['texte_previsualisation'].str.lower()
    df['texte_complet'] = df['texte_complet'].str.lower()
    df['mots_cles_lien'] = df['mots_cles_lien'].str.lower()

    # Supprimer les stopwords
    stop_words = set(stopwords.words('french'))
    df[['titre', 'texte_previsualisation', 'texte_complet', 'mots_cles_lien']] = df[
        ['titre', 'texte_previsualisation', 'texte_complet', 'mots_cles_lien']
    ].map(lambda x: ' '.join([mot for mot in x.split() if mot not in stop_words]))

    # Lemmatisation
    lemmatizer = WordNetLemmatizer()
    df[['titre', 'texte_previsualisation', 'texte_complet', 'mots_cles_lien']] = df[
        ['titre', 'texte_previsualisation', 'texte_complet', 'mots_cles_lien']
    ].map(lambda x: ' '.join([lemmatizer.lemmatize(mot) for mot in x.split()]))

    # Normalisation des données
    df[['titre', 'texte_previsualisation', 'texte_complet', 'mots_cles_lien']] = df[
        ['titre', 'texte_previsualisation', 'texte_complet', 'mots_cles_lien']
    ].map(lambda x: re.sub(r'\S+@\S+', 'EMAIL', x))  # Remplace les emails

    df[['titre', 'texte_previsualisation', 'texte_complet', 'mots_cles_lien']] = df[
        ['titre', 'texte_previsualisation', 'texte_complet', 'mots_cles_lien']
    ].map(lambda x: re.sub(r'\d+', 'NOMBRE', x))  # Remplace les nombres

    df[['titre', 'texte_previsualisation', 'texte_complet', 'mots_cles_lien']] = df[
        ['titre', 'texte_previsualisation', 'texte_complet', 'mots_cles_lien']
    ].map(lambda x: re.sub(r'[:€%$!?,()]', '', x))  # Supprime les caractères spéciaux

    # Supprimer le texte spécifique
    df["texte_complet"] = df["texte_complet"].str.replace(r'\bagence ecofin\b\s*', '', regex=True)

    # Supprimer les traits d'union entre les mots
    df = df.apply(lambda col: col.map(lambda x: x.replace('-', ' ') if isinstance(x, str) else x))

    return df

df=preprocess_data(df)


#-----------------------------------------------#

# Convertir les données en format texte pour les indexer
# Liste pour stocker les documents
documents = []

# Boucle sur les lignes du DataFrame
for index, row in df.iterrows():
    content = f" Titre: {row['titre']}, Texte_previsualisation: {row['texte_previsualisation']}, Texte_complet: {row['texte_complet']}, Lien:{row['mots_cles_lien']}"
    documents.append(Document(page_content=content))

#----------------------------------------------------------------------------#


#FAISS: (Facebook AI Similarity Search) est une bibliothèque conçue pour rechercher efficacement
#  des vecteurs similaires dans de grandes bases de données en utilisant des méthodes d'approximation.
#  Elle est couramment utilisée pour des tâches comme la recherche dans des embeddings
#  ou le regroupement de données à haute dimension

# Indexer les documents avec FAISS
def create_vector_store(documents, embeddings):
    """Crée un vecteur de base de données FAISS à partir des documents."""
    return FAISS.from_documents(documents, embeddings)

#-------------------------------------------------------------------------#

# ** Fonction qui permet de Recherche les documents les plus similaires à la requête et affiche les résultats
def search_and_display_results(vector_store, query, k=2):
    """
    Recherche les documents les plus similaires à la requête et affiche les résultats.

    Args:
        vector_store: La base de données vectorielle (par exemple, FAISS).
        query (str): La requête de recherche.
        k (int): Le nombre de résultats à retourner (par défaut 2).
    """
    # Recherche des documents les plus similaires
    results = vector_store.similarity_search(query, k=k)
    return results

#----------------------------------------------------------------------------------#


# Initialiser les embeddings avec un modèle HuggingFace
from langchain_huggingface import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)


#----------------------------------------------------------------------------------#

#Ce code en dessous définit un modèle de prompt (PromptTemplate) qui structure 
#une entrée en combinant un contexte et une question pour générer une requête lisible 
#et prête à être utilisée dans un modèle de langage.


# Définir le template
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="Contexte : {context}\n\nRépondez à la question suivante : {question}"
)

#-------------------------------------------------------------------------------------#

### Fonction pour interroger le chatbot
def chatbot_faiss(question):
    # Récupération des documents similaires

    docs = create_vector_store(documents, embeddings).similarity_search(question, k=2)
    context = "\n\n".join([doc.page_content for doc in docs])  # Concaténer les documents
    
    # Générer le prompt
    prompt_text = prompt_template.format(context=context, question=question)

    # Appeler le modèle de langage
    response = model.generate_content(prompt_text)

    return response.text



"""


def search_web(question):
    # Implémentation pour rechercher sur le web, ex. utiliser une API comme Google Search ou Bing
    return "Résultat trouvé sur le web pour votre question."  # Exemple statique

import streamlit as st

# Fonction pour interroger le chatbot avec recherche alternative
def chatbot_faiss(question):
    # Récupération des documents similaires
    docs = create_vector_store(documents, embeddings).similarity_search(question, k=2)
    
    if docs:  # Si des documents similaires sont trouvés
        context = "\n\n".join([doc.page_content for doc in docs])  # Concaténer les documents
        # Générer le prompt
        prompt_text = prompt_template.format(context=context, question=question)
    else:  # Si aucun document similaire n'est trouvé
        # Générer un prompt avec un contexte par défaut ou une mention explicite
        st.warning("Aucun contexte pertinent trouvé dans la base vectorisée. Recherche alternative en cours...")
        context = "Aucun contexte pertinent trouvé. Utilisation d'une source externe."
        prompt_text = prompt_template.format(context=context, question=question)
        
        # Rechercher une réponse via une autre source (par exemple, recherche web)
        external_response = search_web(question)
        if external_response:
            return external_response  # Retourner la réponse de la source externe

    # Appeler le modèle de langage avec le contexte ou la mention
    response = model.generate_content(prompt_text)

    return response.text
"""