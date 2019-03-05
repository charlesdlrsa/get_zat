#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Importation des librairies
import matplotlib.pyplot as plt
from browser import vector_request


def compute_precision_recall(questions, request_type, index_inv, path_common_words, collection_tokens,
                             ponderation, answers, threshold=0.4):
    """
    Calcule la précision et le rappel de notre moteur de recherche
    """
    nb_vp = 0
    nb_fp = 0
    nb_fn = 0

    returned_docs = {}
    for index, query in questions.items():
        returned_docs[index] = vector_request(query[0], request_type, index_inv, path_common_words,
                                              collection_tokens, ponderation, threshold=threshold)

        if index in answers:  # on évalue pas sur les documents qui n'ont pas de labels
            for doc_id_pred in returned_docs[index]:
                if doc_id_pred in answers[index]:
                    nb_vp += 1
                else:
                    nb_fp += 1
            for doc_id_real in answers[index]:
                if doc_id_real not in returned_docs[index]:
                    nb_fn += 1

    print(nb_vp, nb_fp, nb_fn)
    precision = nb_vp / (nb_fp + nb_vp) if nb_fp + nb_vp != 0 else 0
    recall = nb_vp / (nb_vp + nb_fn) if nb_vp + nb_fn != 0 else 0

    return precision, recall


def display_graph_pr():
    """
    Cette fonction trace le graphe précision-rappel
    """
    pass
