#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Importation des librairies
import re
import unicodedata
from nltk import word_tokenize
from nltk.corpus import stopwords
from num2words import num2words
from nltk.stem.snowball import FrenchStemmer
from nltk.stem import LancasterStemmer, WordNetLemmatizer
from french_lefff_lemmatizer.french_lefff_lemmatizer import FrenchLefffLemmatizer



class PreProcesseur:

    """
    Classe permettant de pré-processer les données textuelles pour l'analyse de sentiment
    """

    def __init__(self):
        self._lang = 'fr'

    # Méthodes principales de la classe
    def tokenisation(self, sentence):
        """Convert apostrophe (for french) and tokenize sentence"""
        if self._lang == 'fr':
            sentence = sentence.replace("'", " ")  # en français on remplace les apostrophes par des espaces
            return word_tokenize(sentence, language='french')
        else:
            return word_tokenize(sentence)

    def normalize(self, words, verbose=False):
        """Normalize words from list of tokenized words"""
        words = self._remove_non_ascii(words)
        words = self._to_lowercase(words)
        words = self._remove_punctuation(words)
        if verbose:
            print("After removing punctuation : {}".format(words))
        words = self._replace_numbers(words)
        if verbose:
            print("After replacing numbers : {}".format(words))
        words = self._remove_stopwords(words)
        if verbose:
            print("After removing stopwords : {}".format(words))
        return words

    def stem_words(self, words):
        """Stem words in list of tokenized words"""
        if self._lang == 'fr':
            stemmer = FrenchStemmer()
        else:
            stemmer = LancasterStemmer()
        stems = []
        for word in words:
            stem = stemmer.stem(word)
            stems.append(stem)
        return stems

    def lemmatize_verbs(self, words):
        """Lemmatize verbs in list of tokenized words"""
        if self._lang == 'fr':
            lemmatizer = FrenchLefffLemmatizer()
        else:
            lemmatizer = WordNetLemmatizer()
        lemmas = []
        for word in words:
            lemma = lemmatizer.lemmatize(word, pos='v')
            lemmas.append(lemma)
        return lemmas

    # Méthodes privées
    @staticmethod
    def _remove_non_ascii(words):
        """Remove non-ASCII characters from list of tokenized words"""
        new_words = []
        for word in words:
            new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
            new_words.append(new_word)
        return new_words

    @staticmethod
    def _to_lowercase(words):
        """Convert all characters to lowercase from list of tokenized words"""
        new_words = []
        for word in words:
            new_word = word.lower()
            new_words.append(new_word)
        return new_words

    @staticmethod
    def _remove_punctuation(words):
        """Remove punctuation from list of tokenized words"""
        new_words = []
        for word in words:
            new_word = re.sub(r'[^\w\s]', '', word)
            if new_word != '':
                new_words.append(new_word)
        return new_words

    @staticmethod
    def _replace_numbers(words):
        new_words = []
        for word in words:
            if word.isdigit():
                new_word = num2words(float(word), lang='fr')
                new_words.append(new_word)
            else:
                new_words.append(word)
        return new_words

    @staticmethod
    def _remove_stopwords(words):
        """Remove french stop words from list of tokenized words"""
        stop_words = set(stopwords.words('french'))
        not_stopwords = ['n', 'pas', 'ne']
        new_words = []
        for word in words:
            if word not in stop_words or word in not_stopwords:
                new_words.append(word)
        return new_words

    @property
    def lang(self):
        return self._lang


if __name__ == '__main__':

    """ Exemple d'utilisation """

    phrase = "Salut, j'ai joué au foot avec toi hier. On était 10 au total et j'ai marqué 2 but ! " \
             "Tu n'as pas voulu continuer il faisait trop froid !"
    print("Initial : {}".format(phrase))

    preprocess = PreProcesseur()
    w_tok = preprocess.tokenisation(phrase)
    print("\nafter tokenisation : {}".format(w_tok))
    w_norm = preprocess.normalize(w_tok, verbose=True)
    print("\nafter normalization : {}".format(w_norm))

    stems = preprocess.stem_words(w_norm)
    lemmas = preprocess.lemmatize_verbs(w_norm)

    print("\nstems : {}".format(stems))
    print("\nlemmas : {}".format(lemmas))
