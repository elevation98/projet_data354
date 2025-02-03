# Mon Super Projet 🚀

mise en place d'un construire un agent conversationnel capable de répondre aux  
 questions sur des documents spécifiques (Chatbot)

## Description
Mon Super Projet est une application qui utilise l'IA pour répondre  
 à des questions à partir d'une base vectorielle.

## Installation
1. Clonez le dépôt :
   ```bash
   git clone https://github.com/username/project.git

--------------

## Utilisation
 Accédez à l'application sur http://localhost:8501.
 Posez une question ou utilisez l'entrée vocale.
 Obtenez une réponse basée sur une base de données vectorisée.

## Fonctionnalités

* 🎤 Reconnaissance vocale.  
* 🤖 Réponses avec IA.  
* 🔍 Recherche web en cas de contexte manquant.



## Technologies
*Python*  
*Streamlit*  
*FAISS*
*GOOGLE_GEMINI*  

## Architecture
- **Frontend** : Interface construite avec Streamlit.
- **Backend** : Modèle IA pour traiter les requêtes.
- **Base vectorielle** : FAISS pour la recherche.

## 💁 Contribuer
1. Forkez le projet.
2. Créez une branche pour votre fonctionnalité (`git checkout -b feature/nouvelle-fonctionnalite`).
3. Faites vos modifications et validez (`git commit -m "Ajout de ..."`).
4. Envoyez une pull request.


-------------

# Licence
Ce projet est sous licence MIT.

## NB

Les fonctionnalité comme 🔍 et 🎤 ne sont pas actives.  


# Demonstration

Pour exécuter une demo, procedez comme suit:  

1. Clonez ce dépôt :  
`        `  
2. Accédez au dossier de démonstration souhaité :  
`cd PS C:\Users\user\Desktop\Data354> `  

3. Installez les dépendances requises :  
`pip install -r requirements.text`  

4. Créez un fichier basé sur le fichier fourni : modifiez le fichier si nécessaire pour inclure les clés API ou les paramètres de configuration nécessaires..env.env.example  

`cp .env.example .env`  

`.env`  

5. Exécutez l’application Chainlit en mode montre :  
 `streamlit run stream.py `  


L’interface utilisateur de votre chatbot de démonstration   devrait maintenant être opérationnelle dans votre navigateur !  


