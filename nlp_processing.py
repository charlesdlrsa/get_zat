#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Importation des librairies
from nltk import wordpunct_tokenize
from functools import reduce
import os
import re


def tokenisation(collection: dict, path_common_words: str, stemming=False):
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
    for docid, sentence in collection.items():
        sent_token = wordpunct_tokenize(sentence)
        new_sent_token = []
        for word in sent_token:
            word = word.lower()
            word = re.sub(r'[^\w\s]', '', word)
            if word != '' and word not in list_stop_words and not word.isdigit():
                if stemming:
                    from nltk.stem.snowball import SnowballStemmer
                    stemmer = SnowballStemmer("english")
                    word = stemmer.stem(word)
                new_sent_token.append(word)

        collection_tokens[docid] = new_sent_token

    return collection_tokens


def nb_tokens(collection_tokens: dict):
    """
    Cette fonction compte le nombre de tokens dans une collection donnee
    """
    return reduce(lambda acc, y: acc + len(collection_tokens[y]), collection_tokens, 0)

