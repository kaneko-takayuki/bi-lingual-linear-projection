# -*- coding: utf-8 -*-

import argparse
import chainer
import numpy as np
from chainer import Variable
from dto.conf import ConfFFNN

MODEL_DIR = 'trainedModels/'


def init_conf(n_in, n_mid, n_out, batchsize, gpu=-1):
    return ConfFFNN(n_in, n_mid, n_out, batchsize, gpu)


def train(conf: ConfFFNN, train_x: list, train_y: list):
    # ランダム配列作成し、それに沿って学習データリストを並び替え
    n = len(train_x)
    perm = list(np.random.permutation(n))
    random_train_x = [train_x[i] for i in perm]
    random_train_y = [train_y[i] for i in perm]

    sum_loss = 0

    # batchsize個ずつ抜き出し学習
    for i in range(0, n - conf.batchsize, conf.batchsize):
        # ジェネレートされたデータを学習できる状態に持っていく(Variable化)
        sliced_train_x = random_train_x[i:i + conf.batchsize]
        sliced_train_y = random_train_y[i:i + conf.batchsize]
        variable_train_x = Variable(conf.xp.asarray(sliced_train_x).astype(conf.xp.float32))
        variable_train_y = Variable(conf.xp.asarray(sliced_train_y).astype(conf.xp.float32))

        # 学習
        with chainer.using_config('train', True):
            conf.model.cleargrads()
            loss = conf.model(variable_train_x, variable_train_y)
            sum_loss += loss.data
            loss.backward()
            conf.optimizer.update()

    print('loss: ' + str(sum_loss))

    return conf


def main(corpus):
    n_in = 300
    n_mid = 300
    n_out = 300
    batchsize = 30
    gpu = 0

    epochs = 300

    conf = ConfFFNN(n_in, n_mid, n_out, batchsize, gpu)

    en_train_vec = []
    ja_train_vec = []

    with open(corpus, 'r') as f:
        for line in f:
            str_en_vec, str_ja_vec = line.split('\t')
            en_vec = list(map(float, str_en_vec.split(',')))
            ja_vec = list(map(float, str_ja_vec.split(',')))
            en_train_vec.append(en_vec)
            ja_train_vec.append(ja_vec)

    # startエポック〜endエポックまで学習を行う
    for e in range(1, epochs + 1):
        print("epoch: " + str(e))

        # 学習する
        conf = train(conf=conf, train_x=en_train_vec, train_y=ja_train_vec)

        # 保存する
        save_file = MODEL_DIR + "epoch" + str(e) + ".npz"
        ConfFFNN.write(save_file, conf)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='get word2vec japanese word')
    parser.add_argument('--corpus', '-c', type=str, help='source domain corpus(english only now)')
    args = parser.parse_args()

    main(args.corpus)
