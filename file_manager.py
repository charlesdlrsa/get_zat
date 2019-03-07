#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Importation des librairies
import os


def read_file(path: str, tags: list):
    """
    Fonction permettant de lire le fichier en extrayant uniquement certaines balises
    """
    if not os.path.exists(path):
        raise FileNotFoundError("file {} does not exists".format(path))
    collection = {}
    doc_id = 0
    continue_read = True
    change_marqueur = False
    with open(path, 'r') as file:
        for l in file.readlines():

            if l[0] == ".":  # on est face à un marqueur
                marqueur = l[:2]
                if marqueur == ".I":
                    doc_id += 1
                    collection[doc_id] = ""
                    continue
                elif marqueur in tags:
                    continue_read = True
                    change_marqueur = True
                    continue
                else:
                    continue_read = False
                    continue

            if continue_read:
                collection[doc_id] += l.strip()
                if change_marqueur:
                    collection[doc_id] += ". "
                    change_marqueur = False
                else:
                    collection[doc_id] += " "

    return collection


def read_answers(path: str):
    """
    Fonction permettant de lire le fichier en extrayant uniquement les doc_ids
    """
    if not os.path.exists(path):
        raise FileNotFoundError("file {} does not exists".format(path))

    collection = {}
    with open(path, 'r') as file:
        for l in file.readlines():
            split = l.split(' ')
            index, doc_id = int(split[0]), int(split[1])
            if not collection.get(index):
                collection[index] = [doc_id]
            else:
                collection[index] += [doc_id]

    return collection


