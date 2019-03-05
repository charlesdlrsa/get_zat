#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Importation des librairies
import os
from datetime import datetime

from file_manager import read_file, read_answers
from evaluation import compute_precision_recall
from nlp_processing import tokenisation, nb_tokens

from browser import build_index_inv, graphe_frequence_rang, boolean_request, compute_similarity
from vectorizers import BooleanVectorizer, TfIdfVectorizer, FreqNormVectorizer


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

    # Création d'une requête de test
    query = {0: 'what is the language of France ?'}

    # Tokenisation
    collection_tokens = tokenisation(collection, path_common_words, stemming=True)
    query_tokens = tokenisation(query, path_common_words, stemming=True)

    print(collection_tokens[39])
    # ['secant', 'method', 'simultan', 'nonlinear', 'equat', 'procedur', 'simultan', 'solut', 'system', 'necessarili',
    #  'linear', 'equat', 'general', 'secant', 'method', 'singl', 'function', 'variabl']

    print(nb_tokens(collection_tokens))
    # 108113

    index_inv = build_index_inv(collection_tokens)
    print(len(index_inv))
    # 5405

    print(index_inv['method'])
    # {16: 0.25, 26: 0.25, 28: 0.3333333333333333, 35: 0.2, 39: 0.1111111111111111, 42: 0.2, 52: 0.1,
    #  82: 0.05263157894736842, 87: 0.14285714285714285, 88: 0.1111111111111111, ........]

    # graphe_frequence_rang(index_inv, collection_tokens)

    # print(boolean_request("language", "AND", "Implementation", index_inv))

    vectorizer = BooleanVectorizer()
    vec_collections = vectorizer.fit_transform(index_inv, collection_tokens)
    vec_query = vectorizer.transform(query_tokens)

    result = compute_similarity(vec_query, vec_collections, threshold=0.5)
    print(result)

    # precision, recall = compute_precision_recall(questions, 'W', index_inv, path_common_words, collection_tokens,
    #                                             'tfidf', answers, threshold=0.15)
    # print('precision : {}'.format(precision))
    # print('rappel : {}'.format(recall))

    # precision = compute_precision(questions, 'W', index_inv, path_common_words, collection_tokens, 'tfidf', answers)
    # print(precision)


