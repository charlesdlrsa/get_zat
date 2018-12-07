#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Importation des librairies
from nltk import wordpunct_tokenize
import os
import re


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


if __name__ == '__main__':

    path = os.path.join('cacm', 'cacm.all')
    files = read_file(path)
    col_token = tokenisation(files)
    for docid in col_token:
        print(docid, col_token[docid])



