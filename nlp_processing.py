#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Importation des librairies
from nltk import wordpunct_tokenize
from functools import reduce
import os
import re


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


def nb_token(collection_tokens):
    """
    Cette fonction compte le nombre de tokens dans une collection donnee
    """
    return reduce(lambda acc1, y: acc1 + reduce(lambda acc2, z: acc2 + len(z), collection_tokens[y], 0), collection_tokens, 0)

