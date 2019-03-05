# Get_zat : notre projet de recherche d'informations

_Auteurs : Erwan de Kergaradec et Charles de la Roche Saint André_

## Introduction

Vous trouverez dans ce `ReadME` l'ensemble de nos résultats concernant notre projet scolaire de "Fondements à la recherche d'informations". Ce projet scolaire s'appuie sur le dataset CACM contenant un ensemble de documents, questions et réponses. <br>
Chacune des parties détaille le contenu de nos fichiers `.py` ainsi que les rôles respectifs de nos fonctions. <br>
La dernière partie explique comment lancer notre code pour retrouver les mêmes résultats et faire vos propres recherches sur le corpus CACM.

## Lecture des datasets

Toutes les fonctions de lecture des datasets sont présentes dans notre fichier `file_manager.py`. <br>
Elles nous permettent de lire nos datasets de documents, questions et réponses associées et de stocker les informations sous forme de dictionnaires python dont : 
- les clefs sont les identifiants des documents, questions et réponses
- les valeurs sont les données textuelles associées à chaque document, question, réponse

Voici les différentes informations recoltées sur les collections :
- collection de documents : 3204 documents
- collection de questions : 64 questions
- collection de réponses : 52 réponses

## Traitement linguistique

Les fonctions filtrant nos données linguistiques se situent dans le fichier `nlp_processing.py`. <br>
Elles nous permettent de : 
- tokeniser nos textes
- supprimer les chiffres
- supprimer les mots communs
- lémantiser les mots

Une fois notre collection de documents tokenisée, nous trouvons les informations suivantes :
- nombre de tokens de notre collection : 108 113 tokens
- graphe de la fréquence (f) vs rang (r) pour tous les tokens de la collection : 

## Indexation

La construction de notre index inversé a lieu dans le fichier `browser.py`. <br>
Notre index inversé est un dictionnaire dont :
- les clefs sont les tokens de notre collection de documents
- les valeurs sont des dictionnaires dont:
  - les clefs sont les identifiants de nos documents contentant le token
  - les valeurs sont les fréquences d'apparition du token dans le document en question
  
Nous avons pu déterminer de cet index inversé la taille de notre vocabulaire : **5405 tokens**.
 

## Moteur de recherche booléen

Nous avons mis en place un modèle de recherche booléen dans le fichier `browser.py`. <br>
Ce modèle de recherche se base sur une requête contenant deux mots clefs et un opérateur. <br>
Assez basiquement, on prétraite nos deux mots de la même manière que notre collection de documents, puis on regarde quels documents contiennent le premier mot et quels documents contiennent le second mot. <br>
Ensuite, en fonction de l'opérateur (AND, OR, NOT), on renvoie la liste triée de documents correspondants à notre requête booléenne.

## Modèle de recherche vectoriel

### Pondération booléenne

### Pondération tf-idf

### Pondération tf-idf normalisée

## Evaluation pour la collection CACM

Précision et rappel

## Expérimentez-vous même !






