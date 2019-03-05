#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Importation des librairies
import os
from datetime import datetime
from file_manager import read_file, read_answers
from nlp_processing import tokenisation, nb_tokens
from browser import build_index_inv, graphe_frequence_rang, boolean_request


if __name__ == '__main__':

    path_cacm = os.path.join('cacm', 'cacm.all')
    path_questions = os.path.join('cacm', 'query.text')
    path_answers = os.path.join('cacm', 'qrels.text')
    path_common_words = os.path.join('cacm', 'common_words')

    collection = read_file(path_cacm, [".T", ".W", ".K"])
    print(collection[39])
    # "The Secant Method for Simultaneous Nonlinear Equations. A procedure for the simultaneous solution. of a system of not-necessarily-linear equations, a generalization of the secant method for a single function of one variable, is given."

    questions = read_file(path_questions, [".W"])

    answers = read_answers(path_answers)

    collection_tokens = tokenisation(collection, path_common_words, stemming=True)
    print(collection_tokens[39])
    # ['secant', 'method', 'simultan', 'nonlinear', 'equat', 'procedur', 'simultan', 'solut', 'system', 'necessarili', 'linear', 'equat', 'general', 'secant', 'method', 'singl', 'function', 'variabl']

    print(nb_tokens(collection_tokens))
    # 108113

    index_inv = build_index_inv(collection_tokens)
    print(len(index_inv))
    # 5405
    print(index_inv['method'])
    # {16: 0.25, 26: 0.25, 28: 0.3333333333333333, 35: 0.2, 39: 0.1111111111111111, 42: 0.2, 52: 0.1, 82: 0.05263157894736842, 87: 0.14285714285714285, 88: 0.1111111111111111, ........]

    graphe_frequence_rang(index_inv, collection_tokens)

    print(boolean_request("language", "AND", "Implementation", index_inv))

    # doclist = vector_request('What articles exist which deal with TSS (Time Sharing System), '
    #                          'an operating system for IBM computers?',
    #                          'T', index_inv, path_common_words, collection_tokens,
    #                          'tfidf')
    # >>> tfidf : time 8.89s
    #doclist_norm = vector_request('what is the language of France ?', 'W', index_inv, path_common_words,
    #                              collection_tokens, 'tfidf_norm')
    # >>> tfidf_norm : time 13.63s
    #doc_list_f = vector_request('what is the language of France ?', 'W', index_inv, path_common_words,
    #                            collection_tokens, 'freq_norm')
    # >>> freq_norm : time 14.3s
    # print('boolean request : ', result)
    # print('vector request : ', doclist)
    # print('vector request with norm : ', doclist_norm)
    # print('vector request with freq : ', doc_list_f)

    # precision = compute_precision(questions, 'W', index_inv, path_common_words, collection_tokens, 'tfidf', answers)
    # print(precision)


