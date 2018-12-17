#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Importation des librairies
from nltk import wordpunct_tokenize
import os
import re
from functools import reduce
from datetime import datetime
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
    rangs = []
    itr = 0
    for mot in index_inv:
        frequences.append(len(index_inv[mot].keys())/len(collection_tokens))
        rangs.append(itr) # aucune idee de ce qu'est le rang
        itr += 1

    plt.xlabel("Rang du token")
    plt.ylabel("Fréquence du token")
    plt.plot(rangs, frequences)
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
    index_inv = build_index_inv(collection_tokens)
    # print(len(index_inv))
    # >>> 9818
    graphe_frequence_rang(index_inv, collection_tokens)
    result = boolean_request("language", "AND", "system", index_inv)
    print(result)

