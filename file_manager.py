#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Importation des librairies
import os


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
