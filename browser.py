#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Importation des librairies
import os
from math import log, sqrt
import matplotlib.pyplot as plt

from file_manager import read_file
from nlp_processing import tokenisation, nb_token


def build_index_inv(collection_tokens):
    """
    Cette fonction construit un index inverse a partir d'une collection de tokens
    """
    dic_sent = {0: 'T', 1: 'W', 2: 'K'}
    index_inv = {}
    for docid in collection_tokens:
        for i, el in enumerate(collection_tokens[docid]):
            for words in el:
                long = len(collection_tokens[docid][i])
                dic_term = {'T': 1 / long if i == 0 else 0,
                            'W': 1 / long if i == 1 else 0,
                            'K': 1 / long if i == 2 else 0,
                            }
                if words not in index_inv:
                    index_inv[words] = {docid: dic_term}
                else:
                    if docid in index_inv[words]:
                        index_inv[words][docid][dic_sent[i]] += 1/long
                    else:
                        index_inv[words][docid] = dic_term

    return index_inv


def graphe_frequence_rang(index_inv, collection_tokens):
    """
    Trace le graphe de la frequence des termes en fonction de leur rang
    """

    frequences = []
    for mot in index_inv:
        frequences.append(len(index_inv[mot].keys())/len(collection_tokens))

    frequences = sorted(frequences, reverse=True)
    log_frequences = list(map(lambda el: log(el), frequences))
    rangs = range(1, len(frequences) + 1)
    log_rangs = list(map(lambda el: log(el), rangs))

    plt.figure(figsize=(15, 6))
    # Tracé du graphe de la fréquence en fonction du rang du token
    plt.subplot(1, 2, 1)
    plt.xlabel("Rang du token")
    plt.ylabel("Fréquence du token")
    plt.title("Graphe de la fréquence en fonction du rang")
    plt.plot(rangs, frequences)

    # Tracé du graphe du log
    plt.subplot(1, 2, 2)
    plt.xlabel("Log du rang du token")
    plt.ylabel("Log de la fréquence du token")
    plt.title("Graphe du log de la fréquence en fonction du log du rang")
    plt.plot(log_rangs, log_frequences)

    plt.show()


def vectorization_bool(req_term, request_type, index_inv, collection_tokens):
    """
    Vectorise la requête et le corpus selon une méthode booléenne
    """

    q = []
    d = [[0 for i in range(len(index_inv))] for i in range(len(collection_tokens))]
    i = 0
    for term in index_inv:
        if term in req_term:
            q.append(1)
        else:
            q.append(0)

        liste = [key for key in index_inv[term] if index_inv[term][key][request_type] != 0]
        for k in liste:
            d[k-1][i] = 1
        i += 1

    return q, d


def vectorization_tfidf(req_term, request_type, index_inv, collection_tokens):
    """
    Vectorise la requête et le corpus selon la méthode TF-IDF
    """

    nb_doc = len(collection_tokens)
    q = []
    d = [[0 for i in range(len(index_inv))] for i in range(nb_doc)]
    i = 0

    map_request_type = {'T': 0, 'W': 1, 'K': 2}

    for term in index_inv:
        liste = [key for key in index_inv[term] if index_inv[term][key][request_type] != 0]
        idf = log(nb_doc / len(liste), 10) if len(liste) != 0 else 0

        tf_q = req_term.count(term)
        if tf_q != 0:
            q.append((1 + log(tf_q, 10)) * idf)
        else:
            q.append(0)

        for k in liste:
            nb_mots_doc = len(collection_tokens[k][map_request_type[request_type]])
            tf = nb_mots_doc * index_inv[term][k][request_type]
            d[k-1][i] = (1 + log(tf, 10)) * idf
        i += 1

    return q, d


def compute_similarity(vec_request, vec_collections):
    """
    Calcule la similarité entre la requête vectorisée et chaque document vectorisé
    """

    def dotproduct(v1, v2):
        return sum((a * b) for a, b in zip(v1, v2))

    def length(v):
        return sqrt(dotproduct(v, v))

    def sim(v1, v2):
        return dotproduct(v1, v2) / (length(v1) * length(v2))

    simil = []
    for i in range(len(vec_collections)):
        simil.append(sim(vec_collections[i], vec_request))

    doc_similarity = list(map(lambda el: (el[0] + 1, el[1]), enumerate(simil)))
    doc_similarity = sorted(doc_similarity, key=lambda el: el[1], reverse=True)
    doc_similarity = list(map(lambda el: el[0], doc_similarity))[:10]

    return doc_similarity


def boolean_request(mot1, op, mot2, index_inv, request_type):
    """
    Cette fonction permet d'effectuer une recherche booleenne a partir d'une collection tokenise
    """
    if request_type not in ('T', 'W', 'K'):
        raise ValueError("type should be in ('T', 'W', 'K'), is {}".format(request_type))

    try:
        docids_mot1 = [key for key in index_inv[mot1] if index_inv[mot1][key][request_type] != 0]
        docids_mot2 = [key for key in index_inv[mot2] if index_inv[mot2][key][request_type] != 0]
    except KeyError as e:
        return e
    docids_request = []
    if op == "AND":
        for elt in docids_mot1:
            if elt in docids_mot2:
                docids_request.append(elt)
    elif op == "OR":
        for elt in docids_mot1:
            docids_request.append(elt)
        for elt in docids_mot2:
            if elt not in docids_request:
                docids_request.append(elt)
    elif op == "NOT":
        for elt in docids_mot1:
            if elt not in docids_mot2:
                docids_request.append(elt)
    else:
        raise ValueError("Vous n'avez pas entré un operateur valide")

    docids_request.sort()

    return docids_request


def vector_request(request, request_type, index_inv, path_common_words, collection_tokens, ponderation):
    """
    Cette fonction permet d'effectuer une recherche vectorisée a partir d'une collection tokenise
    """

    if request_type not in ('T', 'W', 'K'):
        raise ValueError("type should be in ('T', 'W', 'K'), is {}".format(request_type))
    col = {0: [request]}
    req_term = tokenisation(col, path_common_words)[0][0]

    if ponderation == 'bool':
        vec_request, vec_collection = vectorization_bool(req_term, request_type, index_inv, collection_tokens)
    elif ponderation == 'tfidf':
        vec_request, vec_collection = vectorization_tfidf(req_term, request_type, index_inv, collection_tokens)
    else:
        raise ValueError("ponderation is not correct")

    doc_similarity = compute_similarity(vec_request, vec_collection)

    return doc_similarity


