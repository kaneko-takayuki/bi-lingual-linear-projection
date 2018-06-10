# -*- coding: utf-8 -*-

# 6. 5で求めたベクトルについて、コサイン類似度を求める。

import numpy as np
from gensim.models import KeyedVectors
import constants

OUT_PUT_FILE = 'tmpFiles/similar.tsv'
print('word2vec load')
w2v_model = KeyedVectors.load_word2vec_format(constants.ja_word2vec_path, binary=False)
print('word2vec load complete')


def seek_most_similar(vec):
    similar_words = w2v_model.most_similar(positive=[vec], negative=[])
    return similar_words[:10]


def main():
    corpus = 'tmpFiles/output.tsv'
    with open(corpus, 'r') as f, open(OUT_PUT_FILE, 'w') as outputFile:
        for line in f:
            word, str_vec = line.split('\t')
            vec = np.asarray(list(map(float, str_vec.split(','))))
            similar_words = seek_most_similar(vec)
            words = list(map(str, similar_words))

            outputFile.write(word + '\t' + ','.join(words) + '\n')

if __name__ == '__main__':
    main()
