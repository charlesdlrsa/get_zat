#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Importation des librairies
from browser import compute_similarity, compute_vectors


def compute_precision_recall(query_tokens, collection_tokens, index_inv, answers,
                             vec_type, threshold=0.5, vectorize_request=False):
    """
    Calcule la précision et le rappel de notre moteur de recherche entrainé sur le corpus 'collection_tokens'.
    On évalue les requêtes tokenisées 'query_tokens' en fonction du vrai résultat 'answers'
    On peut faire varier les paramètres :
    - 'vec_type': {'boolean', 'tf-idf', 'tf-idf-norm', 'freq-max'}
    - 'threshold': [0, ..., 1]
    - 'vectorize_request': {True, False}
    """
    nb_vp = 0
    nb_fp = 0
    nb_fn = 0

    vec_query, vec_collections = compute_vectors(query_tokens, collection_tokens, index_inv,
                                                 vec_type, vectorize_request)
    query_result = compute_similarity(vec_query, vec_collections, threshold=threshold)

    for index, relevant_doc_ids in query_result.items():

        if index not in answers:  # on évalue pas sur les documents pas présents dans le fichier de réponse
            continue

        for doc_id_pred in query_result[index]:
            if doc_id_pred in answers[index]:
                nb_vp += 1
            else:
                nb_fp += 1
        for doc_id_real in answers[index]:
            if doc_id_real not in query_result[index]:
                nb_fn += 1

    print(nb_vp, nb_fp, nb_fn)
    precision = nb_vp / (nb_fp + nb_vp) if nb_fp + nb_vp != 0 else 0
    recall = nb_vp / (nb_vp + nb_fn) if nb_vp + nb_fn != 0 else 0

    return precision, recall


def compute_other_metrics(precision: int, recall: int, alpha: float):
    """
    Calcule la F-Mesure, E-Mesure, R-Mesure à partir de la précision et du rappel
    """
    e_measure = 1 - 1 / (alpha / precision + (1 - alpha) / recall)
    f_measure = 1 - e_measure

    return f_measure, e_measure


def display_graph_pr():
    """
    Cette fonction trace le graphe précision-rappel
    """
    pass
