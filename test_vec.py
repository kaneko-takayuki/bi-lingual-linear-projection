# -*- coding: utf-8 -*-

# 5. 文書中に出てくる全ての自立語単語について、4で作成したモデルを通して変換を行う

import argparse
import constants
import treetaggerwrapper as ttw
from gensim.models import KeyedVectors
from dto.conf import ConfFFNN
from chainer import Variable

MODEL_DIR = 'trainedModels/'
OUT_PUT_FILE = 'tmpFiles/output.tsv'

# tree-taggerを使って英文を形態素解析する
tagger = ttw.TreeTagger(TAGLANG='en', TAGDIR='/home/kaneko-takayuki/tree-tagger')

# 「名詞/動詞/形容詞/副詞」系統のタグ(参考: http://computer-technology.hateblo.jp/entry/20150824/p1)
pos_tags = ['NN', 'NNS', 'VV', 'VVD', 'VVG', 'VVN', 'VVP', 'VVZ', 'JJ', 'JJR', 'JJS', 'PB', 'PBR', 'PBS']


def extract_self_sufficient_word(sentence):
    # 文章をパース
    parse_results = list(map(lambda x: x.split('\t'), tagger.TagText(sentence)))
    # タグでフィルタリング
    filtered_by_tags_parse_results = filter(lambda result: (len(result) == 3) and (result[1] in pos_tags), parse_results)
    # 原型のみ抽出
    original_words = list(map(lambda result: result[2], filtered_by_tags_parse_results))
    return original_words


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


def main():
    corpus = '/home/kaneko-takayuki/bi-lingualLinearProjection/sourceDomainCorpus/en-bundle.txt'
    n_in = 300
    n_mid = 300
    n_out = 300
    batchsize = 30
    gpu = 0

    e = 299

    conf = ConfFFNN(n_in, n_mid, n_out, batchsize, gpu)

    print('model load')
    load_file = MODEL_DIR + "epoch" + str(e) + ".npz"
    ConfFFNN.read(load_file, conf)
    print('model load complete')

    words = []
    with open(corpus, 'r') as f:
        for line in f:
            original_words = extract_self_sufficient_word(line)
            words.extend(original_words)

    # 重複削除
    unique_words = list(set(words))

    print('word2vec load')
    # ベクトル求める
    unique_vectors = get_en_word2vec(unique_words)
    print('word2vec translate complete')

    print('output begin')
    with open(OUT_PUT_FILE, 'w') as f:
        for word, vec in zip(unique_words, unique_vectors):
            if vec is None:
                continue
            xp_vec = conf.xp.asarray([vec]).astype(conf.xp.float32)
            variable_vec = Variable(xp_vec)
            y = conf.model.fwd(variable_vec).data[0]
            str_y = list(map(str, y))
            f.write(word + '\t' + ','.join(str_y) + '\n')
    print('output end')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='convert source domain word embeddings')
    args = parser.parse_args()
    main()
