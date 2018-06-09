# -*- coding: utf-8 -*-

import argparse
from gensim.models import KeyedVectors
import constants

OUT_PUT_FILE = 'tmpFiles/vectors.tsv'


def get_en_word2vec(en_words):
    print('get_en_word2vec start')
    w2v_model = KeyedVectors.load_word2vec_format(constants.en_word2vec_path, binary=True)
    print('get_en_word2vec model loaded')
    en_vec_list = []
    for word in en_words:
        if w2v_model.__contains__(word):
            en_vec_list.append(w2v_model[word])
        else:
            en_vec_list.append(None)
    print('get_en_word2vec end')
    return en_vec_list


def get_ja_word2vec(ja_words):
    print('get_ja_word2vec start')
    w2v_model = KeyedVectors.load_word2vec_format(constants.ja_word2vec_path, binary=False)
    print('get_ja_word2vec model loaded')
    ja_vec_list = []
    for word in ja_words:
        if w2v_model.__contains__(word):
            ja_vec_list.append(w2v_model[word])
        else:
            ja_vec_list.append(None)
    print('get_ja_word2vec end')
    return ja_vec_list


def main(corpus):
    en_words = []
    ja_words = []
    with open(corpus, 'r') as f:
        for line in f:
            en_word, ja_word, _ = line.split(',')
            en_words.append(en_word)
            ja_words.append(ja_word)

    en_vec_list = get_en_word2vec(en_words)
    ja_vec_list = get_ja_word2vec(ja_words)

    with open(OUT_PUT_FILE, 'w') as f:
        for en_vec, ja_vec in zip(en_vec_list, ja_vec_list):
            if (en_vec is None) or (ja_vec is None):
                continue
            str_en_vec = map(str, en_vec)
            str_ja_vec = map(str, ja_vec)
            f.write(','.join(str_en_vec) + '\t' + ','.join(str_ja_vec) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='get word2vec japanese word')
    parser.add_argument('--corpus', '-c', type=str, help='source domain corpus(english only now)')
    args = parser.parse_args()

    main(args.corpus)
