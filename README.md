# Get_zat : notre projet de recherche d'informations

_Auteurs : Erwan de Kergaradec et Charles de la Roche Saint André_

## Introduction

Vous trouverez dans ce `ReadME` l'ensemble de nos résultats concernant notre projet scolaire de "Fondements à la recherche d'informations". <br>
Chacune des parties détaille le contenu de nos fichiers .py ainsi que les rôles respectifs de nos fonctions. <br>
La dernière partie explique comment lancer notre code pour retrouver les mêmes résultats et faire vos propres recherches sur le corpus CACM.

## Lecture des datasets

Toutes les fonctions de lecture des datasets sont présentes dans notre fichier `file_manager.py`. <br>
Elles nous permettent de lire nos datasets de corpus, questions et réponses associées et de sotcker les informations sous forme de dictionnaires python dont : 
- les clefs sont les identifiants des documents, questions et réponses
- les valeurs sont les données textuelles associées à chaque document, question, réponse

Voici les différentes informations recoltées sur les collections :
- collection de documents : 3000 documents
- collection de questions :  
- collection de réponses : 

## Traitement linguistique

Les fonctions filtrant nos données linguistiques se situent dans le fichier `nlp_processing.py`. <br>
Elles nous permettent de : 
- tokeniser nos textes
- supprimer les chiffres
- supprimer les mots communs
- lémantiser les mots

Une fois notre collection de documents tokenisée, nous trouvons les informations suivantes :
- nombre de tokens de notre collection : 
- taille du vocabulaire de notre collection : 
- taille du vocabulaire pour une collection d'un million de tokens :
- graphe de la fréquence (f) vs rang (r) pour tous les tokens de la collection : 

## Indexation

La construction de notre index inversé a lieu dans le fichier `browser.py`. <br>
Notre index inversé est un dictionnaire dont :
- les clefs sont les tokens de notre collection de documents
- les valeurs sont des dictionnaires dont:
  - les clefs sont les identifiants de nos documents contentant le token
  - les valeurs sont les fréquences d'apparition du token dans le dit-document
 
 

## Moteur de recherche booléen



## Modèle de recherche vectoriel

## Evaluation pour la collection CACM

## Expérimentez-vous même !






