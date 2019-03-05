#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Importation des librairies
from math import log, sqrt
import matplotlib.pyplot as plt
import numpy as np


class BooleanVectorizer:
    """
    Classe permettant de vectoriser un corpus et une requête selon une méthode booléenne
    """

    def __init__(self, request_type):
        self.__request_type = request_type
        self.__inversed_index = []

    def fit_transform(self, inversed_index, collection_tokens):
        """
        Apprend la vectorisation du corpus à partir d'un index inversé et d'une collection de tokens
        """
        self.__inversed_index = inversed_index
        vec_matrix = {doc_id: [0 for _ in range(len(inversed_index))] for doc_id in collection_tokens}
        for i, term in enumerate(inversed_index):
            doc_ids = [doc_id for doc_id in inversed_index[term]
                       if inversed_index[term][doc_id][self.__request_type] != 0]
            for doc_id in doc_ids:
                vec_matrix[doc_id][i] = 1

        return vec_matrix

    def transform(self, requests_term: dict):
        """
        Vectorise une liste de requête à partir de l'apprentissage précédent
        """
        if len(self.__inversed_index) == 0:
            raise InterruptedError("method fit should be call first.")
        vec_request = {}
        for index, query in requests_term.items():
            vec_req = []
            for term in self.__inversed_index:
                if term in query[0]:
                    vec_req.append(1)
                else:
                    vec_req.append(0)
            vec_request[index] = vec_req
        return vec_request


class TfIdfVectorizer:
    """
    Classe permettant de vectoriser un corpus et une requête selon la méthode TF-IDF
    """

    def __init__(self, request_type, norm=False, vectorize_request=False):
        self.__request_type = request_type
        self.__norm = norm
        self.__vectorize_request = vectorize_request
        self.__inversed_index = []
        self.idf = []

    def fit_transform(self, inversed_index, collection_tokens):
        """
         Apprend la vectorisation du corpus à partir d'un index inversé et d'une collection de tokens
        """
        self.__inversed_index = inversed_index
        nb_docs = len(collection_tokens)
        vec_matrix = {doc_id: [0 for _ in range(len(inversed_index))] for doc_id in collection_tokens}

        map_request_type = {'T': 0, 'W': 1, 'K': 2}

        for i, term in enumerate(inversed_index):

            # list of doc_id where term is present
            doc_ids = [key for key in inversed_index[term] if inversed_index[term][key][self.__request_type] != 0]
            idf = log(nb_docs / len(doc_ids), 10) if len(doc_ids) != 0 else 0
            self.idf.append(idf)

            for doc_id in doc_ids:
                nb_mots_doc = len(collection_tokens[doc_id][map_request_type[self.__request_type]])
                tf = nb_mots_doc * inversed_index[term][doc_id][self.__request_type]
                if self.__norm:
                    vec_matrix[doc_id][i] = (1 + log(tf)) * idf
                else:
                    vec_matrix[doc_id][i] = (1 + log(tf, 10)) * idf

        # for normalized tf_idf
        if self.__norm:
            for doc_id in vec_matrix:
                sum_d = sum(vec_matrix[doc_id])
                n_d = 1 / sqrt(sum_d) if sum_d != 0 else 0
                for term in range(len(vec_matrix[doc_id])):
                    vec_matrix[doc_id][term] = n_d * vec_matrix[doc_id][term]

        return vec_matrix

    def transform(self, requests_term: dict):
        """
        Vectorise une liste de requête à partir de l'apprentissage précédent
        """
        if len(self.__inversed_index) == 0:
            raise InterruptedError("method fit should be call first.")
        vec_request = {}
        for index, query in requests_term.items():
            query = query[0]
            vec_req = []
            for i, term in enumerate(self.__inversed_index):
                if self.__vectorize_request:
                    tf_request = len([1 for _ in query if _ == term])
                    vec_req.append((1 + log(tf_request, 10)) * self.idf[i]) if tf_request != 0 else vec_req.append(0)
                else:
                    vec_req.append(1) if term in query else vec_req.append(0)

            if self.__norm and self.__vectorize_request:
                n_d = 1 / sqrt(sum(vec_req)) if sum(vec_req) != 0 else 0
                for term in range(len(vec_req)):
                    vec_req[term] = n_d * vec_req[term]

            vec_request[index] = vec_req
        return vec_request


class FreqNormVectorizer:

    def __init__(self):
        pass

    def fit(self):
        pass

    def transform(self):
        pass