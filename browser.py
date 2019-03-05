#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Importation des librairies
from math import log, sqrt
import matplotlib.pyplot as plt
import numpy as np

from nlp_processing import tokenisation


def build_index_inv(collection_tokens):
    """
    Cette fonction construit un index inversé a partir d'une collection de tokens
    """
    index_inv = {}
    for docid, tokens in collection_tokens.items():
        for token in tokens:
            long = len(collection_tokens[docid])
            if token not in index_inv:
                index_inv[token] = {docid: 1/long}
            else:
                if docid in index_inv[token]:
                    index_inv[token][docid] += 1/long
                else:
                    index_inv[token][docid] = 1/long

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
    d = [[0 for _ in range(len(index_inv))] for _ in range(len(collection_tokens))]
    i = 0
    for term in index_inv:
        # inversed index vectorization
        liste = [key for key in index_inv[term] if index_inv[term][key][request_type] != 0]
        for k in liste:
            d[k-1][i] = 1
        i += 1

        # request vectorization
        if term in req_term:
            q.append(1)
        else:
            q.append(0)

    return q, d


def vectorization_tfidf(req_term, request_type, index_inv, collection_tokens, norm=False, vectorize_request=False):
    """
    Vectorise la requête et le corpus selon la méthode TF-IDF
    Avec l'argument norm, la requête est préalablement normalisée
    """

    nb_docs = len(collection_tokens)
    q = []
    d = [[0 for _ in range(len(index_inv))] for _ in range(nb_docs)]

    map_request_type = {'T': 0, 'W': 1, 'K': 2}

    i = 0
    for term in index_inv:

        # list of doc_id where term is present
        doc_ids = [key for key in index_inv[term] if index_inv[term][key][request_type] != 0]
        idf = log(nb_docs / len(doc_ids), 10) if len(doc_ids) != 0 else 0

        # request vectorization
        if vectorize_request:
            tf_request = len([1 for _ in req_term if _ == term])
            q.append((1 + log(tf_request, 10)) * idf) if tf_request != 0 else q.append(0)
        else:
            q.append(1) if term in req_term else q.append(0)

        for k in doc_ids:
            nb_mots_doc = len(collection_tokens[k][map_request_type[request_type]])
            tf = nb_mots_doc * index_inv[term][k][request_type]
            if norm:
                d[k-1][i] = (1 + log(tf)) * idf
            else:
                d[k-1][i] = (1 + log(tf, 10)) * idf
        i += 1

    # for normalized tf_idf
    if norm:
        for doc_id in range(len(d)):
            sum_d = sum(d[doc_id])
            n_d = 1 / sqrt(sum_d) if sum_d != 0 else 0
            for term in range(len(d[doc_id])):
                d[doc_id][term] = n_d * d[doc_id][term]
        if vectorize_request:
            n_d = 1 / sqrt(sum(q)) if sum(q) != 0 else 0
            for term in range(len(q)):
                q[term] = n_d * q[term]

    return q, d


def vectorization_freq_norm(req_term, request_type, index_inv, collection_tokens):
    """
    Vectorise la requête et le corpus selon la méthode de fréquence normalisée
    """

    nb_doc = len(collection_tokens)
    q = []
    d = [[0 for _ in range(len(index_inv))] for _ in range(nb_doc)]
    i = 0

    map_request_type = {'T': 0, 'W': 1, 'K': 2}

    for term in index_inv:
        # doc_id where term is present
        liste = [key for key in index_inv[term] if index_inv[term][key][request_type] != 0]

        # request vectorization
        if term in req_term:
            q.append(1)
        else:
            q.append(0)

        # inversed index vectorization
        for k in liste:
            nb_mots_doc = len(collection_tokens[k][map_request_type[request_type]])
            tf = nb_mots_doc * index_inv[term][k][request_type]
            d[k - 1][i] = tf

        i += 1

    # normalization per document
    for doc_id in range(len(d)):
        freq_max = max([w for w in d[doc_id]])
        n_d = 1 / sqrt(freq_max) if freq_max != 0 else 0
        for term in range(len(d[doc_id])):
            d[doc_id][term] = n_d * d[doc_id][term]

    return q, d


def compute_similarity(vec_request, vec_collections, threshold=0.2):
    """
    Calcule la similarité entre la requête vectorisée et chaque document vectorisé
    Renvoie la liste des documents dont la similarité est supérieure au seuil
    """

    def sim(v1, v2, norm_v2):
        v1 = np.array(v1)
        v2 = np.array(v2)
        return np.vdot(v1, v2) / (np.linalg.norm(v1) * norm_v2)

    simil_request = {}
    for index, query_vector in vec_request.items():
        norm_query = np.linalg.norm(query_vector)
        relevant_doc_ids = []
        for doc_id, doc_vector in vec_collections.items():
            if sum(doc_vector) != 0 and sim(doc_vector, query_vector, norm_query) > threshold:
                relevant_doc_ids.append(doc_id)
        simil_request[index] = relevant_doc_ids
    return simil_request


def boolean_request(mot1, op, mot2, index_inv, request_type):
    """
    Cette fonction permet d'effectuer une recherche booleenne a partir d'une collection tokenise
    """
    if request_type not in ('T', 'W', 'K'):
        raise ValueError("type should be in ('T', 'W', 'K'), is {}".format(request_type))

    mot1, mot2 = mot1.lower(), mot2.lower()

    docids_mot1 = []
    docids_mot2 = []

    try:
        docids_mot1 = [key for key in index_inv[mot1] if index_inv[mot1][key][request_type] != 0]
        docids_mot2 = [key for key in index_inv[mot2] if index_inv[mot2][key][request_type] != 0]
    except KeyError:
        pass

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
        raise ValueError("You should enter a valid operator")

    docids_request.sort()

    return docids_request


def vector_request(request, request_type, index_inv, path_common_words, collection_tokens, ponderation, threshold=0.8):
    """
    Cette fonction permet d'effectuer une recherche vectorisée a partir d'une collection tokenisée
    On peut choisir la méthode de pondéraion avec l'argument 'ponderation'
    On peut faire varier le seuil de similarité avec l'argument 'threshold'
    """

    if request_type not in ('T', 'W', 'K'):
        raise ValueError("type should be in ('T', 'W', 'K'), is {}".format(request_type))
    col = {0: [request]}
    req_term = tokenisation(col, path_common_words)[0][0]

    # il faudrait vectoriser indépendamment le corpus et la requête pour gagner en efficacité !
    if ponderation == 'bool':
        vec_request, vec_collection = vectorization_bool(req_term, request_type, index_inv, collection_tokens)
    elif ponderation == 'tfidf':
        vec_request, vec_collection = vectorization_tfidf(req_term, request_type, index_inv, collection_tokens)
    elif ponderation == 'tfidf_norm':
        vec_request, vec_collection = vectorization_tfidf(req_term, request_type, index_inv, collection_tokens,
                                                          norm=True)
    elif ponderation == 'freq_norm':
        vec_request, vec_collection = vectorization_freq_norm(req_term, request_type, index_inv, collection_tokens)
    else:
        raise ValueError("ponderation is not correct, should be 'bool', 'tfidf', 'tfidf_norm', 'freq_norm'.")

    doc_similarity = compute_similarity(vec_request, vec_collection, threshold=threshold)

    return doc_similarity





