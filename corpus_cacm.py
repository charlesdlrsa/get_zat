#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Importation des librairies
from nltk import wordpunct_tokenize
import os
import re
from functools import reduce


def read_file(path):
    """
    Fonction permettat de lire le fichier en extrayant uniquement certaines balises
    """
    if not os.path.exists(path):
        raise FileNotFoundError("file {} does not exists".format(path))
    documents = {}
    doc_id = None
    continue_read = True
    change_marqueur = False
    with open(path, 'r') as file:
        for l in file.readlines():

            if l[0] == ".":  # on est face à un marqueur
                marqueur = l[:2]
                if marqueur == ".I":
                    doc_id = int(l[3:].strip())
                    documents[doc_id] = []
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
                    documents[doc_id].append(l.strip() + " ")
                    change_marqueur = False
                else:
                    documents[doc_id][-1] += l.strip() + " "

    return documents


def tokenisation(collection):
    """
    Cette fonction tokenise les mots, enlève la ponctuation et met en minuscule
    """
    collection_token = {}
    for docid in collection:
        doc_token = []
        for sentence in collection[docid]:
            sent_token = wordpunct_tokenize(sentence)
            new_sent_token = []
            for word in sent_token:
                new_word = re.sub(r'[^\w\s]', '', word)
                if new_word != '':
                    new_sent_token.append(new_word.lower())
            doc_token.append(new_sent_token)
        collection_token[docid] = doc_token
    return collection_token


def remove_common_words(token, path_common):

    if not os.path.exists(path_common):
        raise FileNotFoundError("file {} doesn't exist".format(path_common))

    list_stop_words = []
    with open(path_common, 'r') as file:
        for l in file.readlines():
            list_stop_words.append(l[:-1])

    for docid in token:
        for sentence in token[docid]:
            for words in sentence:
                if words in list_stop_words:
                    sentence.remove(words)

    return token


def nb_token(collection):
    return reduce(lambda acc1, y: acc1 + reduce(lambda acc2, z: acc2 + len(z), collection[y], 0), collection, 0)


def agg_terms(collection):
    def mapper(docid):
        dic_term = {'T': collection[docid][0] if len(collection[docid]) > 0 else [],
                    'W': collection[docid][1] if len(collection[docid]) > 1 else [],
                    'K': collection[docid][2] if len(collection[docid]) > 2 else []
                    }
        return docid, dic_term
    map_coll = list(map(mapper, collection))

    print(map_coll[2])

    dic_sent = {0: 'T', 1: 'W', 2: 'K'}
    index_inv = {}
    for docid in collection:
        dic_term = {'T': [],
                    'W': [],
                    'K': []
                    }
        for i, el in enumerate(collection[docid]):
            long = len(collection[docid[i]])
            for words in el:
                dic_term = {'T': 1/len(collection[docid][0]) if i == 0 else 0,
                            'W': 1/len(collection[docid][1]) if i == 1 else 0,
                            'K': 1/len(collection[docid][2]) if i == 2 else 0,
                            }
                if words not in index_inv:
                    index_inv[words] = [(docid, dic_term)]
                else:
                    ancienne_freq = index_inv[words][docid]['T']
                    index_inv[words].append((docid, dic_term))

    exemple = {'techniques': [(3, {'T': 0.2, 'W': 0, 'K': 0}), (16, {'T': 0.1, 'W': 0, 'K': 0})]}


if __name__ == '__main__':

    path = os.path.join('cacm', 'cacm.all')
    path_common = os.path.join('cacm', 'common_words')
    files = read_file(path)
    col_token = tokenisation(files)
    col_token = remove_common_words(col_token, path_common)

    agg_terms(col_token)

    #for docid in col_token:
    #    print(docid, col_token[docid])

    # nombre de tokens


