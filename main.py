#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Importation des librairies
import os

from code.file_manager import read_file, read_answers
from code.nlp_processing import tokenisation, nb_tokens
from code.browser import build_index_inv, display_graph_freq_rank, boolean_request, compute_similarity
from code.vectorizers import BooleanVectorizer, TfIdfVectorizer, FreqNormVectorizer
from code.evaluation import compute_precision_recall, display_graph_pr

if __name__ == '__main__':

    # Définition des chemins vers les données
    path_cacm = os.path.join('cacm', 'cacm.all')
    path_questions = os.path.join('cacm', 'query.text')
    path_answers = os.path.join('cacm', 'qrels.text')
    path_common_words = os.path.join('cacm', 'common_words')

    # Lecture de chaque fichiers
    collection = read_file(path_cacm, [".T", ".W", ".K"])
    questions = read_file(path_questions, [".W"])
    answers = read_answers(path_answers)
    print("Nombre de documents : ", len(collection), "\n")
    print("Nombre de questions : ", len(questions), "\n")
    print("Nombre de réponses : ", len(answers), "\n")
    print("Document n°39 : ", collection[39], "\n")

    # Tokenisation
    collection_tokens = tokenisation(collection, path_common_words, stemming=True)
    print("Nombre de tokens du corpus de tous les documents : ", nb_tokens(collection_tokens), "\n")
    print("Tokens du document 39 : ", collection_tokens[39], "\n")

    # Indexation
    index_inv = build_index_inv(collection_tokens)
    print("Taille de notre vocabulaire : ", len(index_inv), "\n")
    print("Index inversé du mot 'variable' : \n", index_inv['variabl'], "\n")
    print("Fermez la fenêtre du graphe fréquence-rang des tokens pour que le code continue. \n")
    display_graph_freq_rank(index_inv, collection_tokens)

    # Modèle de recherche booléen
    print("Résultat de recherche booléenne sur la requête suivante : 'language', 'AND', 'Implementation' : \n")
    print(boolean_request("language", "AND", "Implementation", index_inv), '\n')

    # Création d'une requête de test
    query = {0: 'What articles exist which deal with TSS (Time Sharing System), an operating system for IBM computers?'}
    query_tokens = tokenisation(query, path_common_words, stemming=True)

    # Modèle de recherche vectoriel (ici on a pris tf-idf mais possibilité d'utiliser les deux autres)
    ## vectorizer = BooleanVectorizer()
    ## vectorizer = FreqNormVectorizer(vectorize_request=False)
    vectorizer = TfIdfVectorizer(norm=False, vectorize_request=False)
    print("On teste un modèle de recherche tf-idf sur une requête test : \n", query[0], '\n')

    # Vectorisation selon la méthode choisie
    vec_collections = vectorizer.fit_transform(index_inv, collection_tokens)
    vec_query = vectorizer.transform(query_tokens)

    # Calcul de la similarité
    result = compute_similarity(vec_query, vec_collections, threshold=0.2)
    print('Calcul de la similarité sur notre requête test : {}'.format(result), '\n')

    # Evaluation du corpus avec la modèle de recherche vectoriel tf-idf
    print("On teste un modèle de recherche tf-idf sur l'ensemble des questions. \n")
    print("(Veuillez patienter, le modèle met un peu de temps à tourner...) \n")
    precision, recall = compute_precision_recall(questions, collection_tokens, index_inv, answers,
                                                 'tf-idf', threshold=0.15, vectorize_request=False)
    print("Precision pour le modele tf-idf sur l'ensemble des questions avec un seuil de 0.15: {}".format(precision), '\n')
    print("Rappel pour le modele tf-idf sur l'ensemble des questions avec un seuil de 0.15: {}".format(recall), '\n')

    # Très long à faire tourner car on calcule la précision et le rappel pour plusieurs tresholds sur tout le corpus
    # print("Fermez la fenêtre du graphe pour que le code continue. \n")
    # display_graph_pr(questions, collection_tokens, index_inv, answers, 'tf-idf', vectorize_request=False)


