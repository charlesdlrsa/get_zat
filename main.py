#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Importation des librairies
import os

from file_manager import read_file
from nlp_processing import tokenisation, nb_token
from browser import build_index_inv, graphe_frequence_rang, boolean_request, vector_request


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
    result = boolean_request("language", "AND", "france", index_inv, 'T')
    doclist = vector_request('what is the language of France ?', 'T', index_inv, path_common_words, collection_tokens,
                             'tfidf')
    print(result)
    print(doclist)

