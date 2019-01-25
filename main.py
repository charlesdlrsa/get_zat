#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Importation des librairies
import os
from datetime import datetime
from file_manager import read_file, read_answers
from nlp_processing import tokenisation, nb_token
from browser import build_index_inv, graphe_frequence_rang, boolean_request, vector_request, compute_precision


if __name__ == '__main__':

    path_cacm = os.path.join('cacm', 'cacm.all')
    path_questions = os.path.join('cacm', 'query.text')
    path_answers = os.path.join('cacm', 'qrels.text')
    path_common_words = os.path.join('cacm', 'common_words')
    collection = read_file(path_cacm, [".T", ".W", ".K"])
    print(len(collection))
    questions = read_file(path_questions, [".W"])
    answers = read_answers(path_answers)
    collection_tokens = tokenisation(collection, path_common_words)
    print(len(collection_tokens))
    # >>> 3s59ms to tokenise
    # print(nb_token(collection_tokens))
    # >>> 118931
    # print(collection_tokens[103])
    # >>> [['cope', 'console'], ['each', year', ..]] # pas de mots clés
    index_inv = build_index_inv(collection_tokens)
    # >>> 0.13s to build index_inv
    # print(len(index_inv))
    # >>> 9723
    # print(index_inv['language'])
    # >>> {1: {'T': 0.2, 'W': 0, 'K': 0}, 82: {'T': 0.16666666666666666, 'W': 0.07692307692307693, 'K': 0}, ..}

    # graphe_frequence_rang(index_inv, collection_tokens)
    # result = boolean_request("language", "AND", "Implementation", index_inv, 'T')
    # >>> boolean : time 0.2ms
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

    precision = compute_precision(questions, 'W', index_inv, path_common_words, collection_tokens, 'tfidf', answers)
    print(precision)


