#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Importation des librairies
from vectorizers import BooleanVectorizer, TfIdfVectorizer, FreqNormVectorizer
from math import log
import matplotlib.pyplot as plt
import numpy as np


def boolean_request(word1, op, word2, index_inv):
    """
    Cette fonction permet d'effectuer une recherche booleenne a partir d'une collection tokenise
    """

    word1, word2 = word1.lower(), word2.lower()
    from nltk.stem.snowball import SnowballStemmer
    stemmer = SnowballStemmer("english")
    word1, word2 = stemmer.stem(word1), stemmer.stem(word2)

    try:
        docids_word1 = list(index_inv[word1].keys())
    except KeyError:
        docids_word1 = []
    try:
        docids_word2 = list(index_inv[word2].keys())
    except KeyError:
        docids_word2 = []

    docids_request = []
    if op == "AND":
        for elt in docids_word1:
            if elt in docids_word2:
                docids_request.append(elt)
    elif op == "OR":
        for elt in docids_word1:
            docids_request.append(elt)
        for elt in docids_word2:
            if elt not in docids_request:
                docids_request.append(elt)
    elif op == "NOT":
        for elt in docids_word1:
            if elt not in docids_word2:
                docids_request.append(elt)
    else:
        raise ValueError("You should enter a valid operator")

    docids_request.sort()
    return docids_request


def build_index_inv(collection_tokens):
    """
    Cette fonction construit un index inversé a partir d'une collection de tokens
    """
    index_inv = {}
    for docid, tokens in collection_tokens.items():
        for token in tokens:
            long = len(tokens)
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


def compute_similarity(vec_request, vec_collections, threshold=0.5):
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


def compute_vectors(query_tokens, collection_tokens, index_inv, vec_type, vectorize_request=False):
    """
    Cette fonction permet d'effectuer une recherche vectorisée a partir d'une collection tokenisée
    On peut choisir la méthode de pondéraion avec l'argument 'ponderation'
    On peut faire varier le seuil de similarité avec l'argument 'threshold'
    """
    if vec_type == 'boolean':
        vectorizer = BooleanVectorizer()
        vec_collections = vectorizer.fit_transform(index_inv, collection_tokens)
        vec_query = vectorizer.transform(query_tokens)
    elif vec_type == 'tf-idf':
        vectorizer = TfIdfVectorizer(norm=False, vectorize_request=vectorize_request)
        vec_collections = vectorizer.fit_transform(index_inv, collection_tokens)
        vec_query = vectorizer.transform(query_tokens)
    elif vec_type == 'tf-idf-norm':
        vectorizer = TfIdfVectorizer(norm=True, vectorize_request=vectorize_request)
        vec_collections = vectorizer.fit_transform(index_inv, collection_tokens)
        vec_query = vectorizer.transform(query_tokens)
    elif vec_type == 'freq-norm':
        vectorizer = FreqNormVectorizer(vectorize_request=vectorize_request)
        vec_collections = vectorizer.fit_transform(index_inv, collection_tokens)
        vec_query = vectorizer.transform(query_tokens)
    else:
        raise ValueError("'vec_type' should be in {'boolean', 'tf-idf', 'tf-idf-norm', 'freq-max'")
    return vec_query, vec_collections





