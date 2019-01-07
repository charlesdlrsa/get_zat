#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Importation des librairies
from nltk import wordpunct_tokenize
import os
import re
from functools import reduce
from math import log, cos, sqrt
import time
import matplotlib.pyplot as plt


def read_file(path):
    """
    Fonction permettat de lire le fichier en extrayant uniquement certaines balises
    """
    if not os.path.exists(path):
        raise FileNotFoundError("file {} does not exists".format(path))
    collection = {}
    doc_id = None
    continue_read = True
    change_marqueur = False
    with open(path, 'r') as file:
        for l in file.readlines():

            if l[0] == ".":  # on est face à un marqueur
                marqueur = l[:2]
                if marqueur == ".I":
                    doc_id = int(l[3:].strip())
                    collection[doc_id] = []
                    continue
                elif marqueur in (".T", ".W", ".K"):
                    continue_read = True
                    change_marqueur = True
                    continue
                else:
                    continue_read = False
                    continue

            if continue_read:
                if change_marqueur:
                    collection[doc_id].append(l.strip() + " ")
                    change_marqueur = False
                else:
                    collection[doc_id][-1] += l.strip() + " "

    return collection


def tokenisation(collection, path_common_words):
    """
    Cette fonction tokenise les mots, enlève la ponctuation, les mots communs et met en minuscule
    """

    if not os.path.exists(path_common_words):
        raise FileNotFoundError("file {} doesn't exist".format(path_common_words))

    list_stop_words = []
    with open(path_common_words, 'r') as file:
        for l in file.readlines():
            list_stop_words.append(l[:-1])

    collection_tokens = {}
    for docid in collection:
        doc_token = []
        for sentence in collection[docid]:
            sent_token = wordpunct_tokenize(sentence)
            new_sent_token = []
            for word in sent_token:
                new_word = re.sub(r'[^\w\s]', '', word)
                if (new_word != '') and (new_word not in list_stop_words):
                    new_sent_token.append(new_word.lower())
            doc_token.append(new_sent_token)
        collection_tokens[docid] = doc_token
    return collection_tokens


# def remove_common_words(collection_tokens, path_common_words):
#     """
#     Cette fonction retire les mots communs de nos tokens.
#     """
#     if not os.path.exists(path_common_words):
#         raise FileNotFoundError("file {} doesn't exist".format(path_common_words))
#
#     list_stop_words = []
#     with open(path_common_words, 'r') as file:
#         for l in file.readlines():
#             list_stop_words.append(l[:-1])
#
#     for docid in collection_tokens:
#         for sentence in collection_tokens[docid]:
#             for words in sentence:
#                 if words in list_stop_words:
#                     sentence.remove(words)
#
#     return collection_tokens


def nb_token(collection_tokens):
    """
    Cette fonction compte le nombre de tokens dans une collection donnee
    """
    return reduce(lambda acc1, y: acc1 + reduce(lambda acc2, z: acc2 + len(z), collection_tokens[y], 0), collection_tokens, 0)


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


def boolean_request(mot1, op, mot2, index_inv):
    """
    Cette fonction permet d'effectuer une recherche booleenne a partir d'une collection tokenise
    """
    try:
        docids_mot1 = list(index_inv[mot1].keys())
        docids_mot2 = list(index_inv[mot2].keys())
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


def vectorization_bool(request, request_type, index_inv, collection_tokens, path_common_words):

    if request_type not in ('T', 'W', 'K'):
        raise ValueError("type should be in ('T', 'W', 'K'), is {}".format(request_type))
    col = {0: [request]}
    req_term = tokenisation(col, path_common_words)[0][0]

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


def vectorization_tfidf(request, request_type, index_inv, collection_tokens, path_common_words):

    if request_type not in ('T', 'W', 'K'):
        raise ValueError("type should be in ('T', 'W', 'K'), is {}".format(request_type))
    col = {0: [request]}
    req_term = tokenisation(col, path_common_words)[0][0]

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


def vector_request_bool(request, request_type, index_inv, path_common_words, collection_tokens, ponderation):

    if ponderation == 'bool':
        vec_request, vec_collection = vectorization_bool(request, request_type, index_inv, collection_tokens, path_common_words)
    elif ponderation == 'tfidf':
        vec_request, vec_collection = vectorization_tfidf(request, request_type, index_inv, collection_tokens, path_common_words)
    else:
        raise ValueError("ponderation is not correct")

    doc_similarity = compute_similarity(vec_request, vec_collection)

    return doc_similarity


if __name__ == '__main__':

    path = os.path.join('cacm', 'cacm.all')
    path_common_words = os.path.join('cacm', 'common_words')
    collection = read_file(path)
    # time1 = datetime.now()
    collection_tokens = tokenisation(collection, path_common_words)
    # time2 = datetime.now()
    # print(time2-time1)
    # >>> 3s749ms
    # print(nb_token(collection_tokens))
    # >>> 118931
    # print(collection_tokens[103])
    # >>> [['cope', 'console'], ['each', year', ..]] # pas de mots clés
    index_inv = build_index_inv(collection_tokens)
    # print(len(index_inv))
    # >>> 9723
    # print(index_inv['language'])
    # >>> {1: {'T': 0.2, 'W': 0, 'K': 0}, 82: {'T': 0.16666666666666666, 'W': 0.07692307692307693, 'K': 0}, ..}

    # graphe_frequence_rang(index_inv, collection_tokens)
    # result = boolean_request("language", "AND", "system", index_inv)
    doclist = vector_request_bool('what is the language of France ?', 'T', index_inv, path_common_words, collection_tokens, 'tfidf')
    print(doclist)

