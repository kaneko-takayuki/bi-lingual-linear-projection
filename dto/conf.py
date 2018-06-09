# -*- coding: utf-8 -*-

from abc import ABCMeta
import numpy as np
from chainer import cuda
from chainer import optimizers
from chainer import serializers
from model.ffnn import FFNN


class ModelBase(metaclass=ABCMeta):
    """
    モデルに共通する処理(保存、読み込み)を備えた抽象クラス
    """
    @staticmethod
    def write(file_name, conf):
        """
        モデルを保存する
        :param file_name: 保存するファイル名
        :param conf: ディープラーニング設定ファイル
        :return: なし
        """
        # CPUモードで統一的に保存する
        conf.model.to_cpu()
        serializers.save_npz(file_name, conf.model)

        # GPU設定
        if conf.gpu >= 0:
            cuda.get_device_from_id(conf.gpu).use()
            cuda.check_cuda_available()
            conf.model.to_gpu()

    @staticmethod
    def read(file_name, conf):
        """
        モデルを読み込む
        :param file_name: 読み込むファイル名 
        :param conf: モデルの読み込み先設定オブジェクト
        :return: 
        """
        # モデルをロードする
        serializers.load_npz(file_name, conf.model)

        # GPU設定
        if conf.gpu >= 0:
            cuda.get_device_from_id(conf.gpu).use()
            cuda.check_cuda_available()
            conf.model.to_gpu()


class ConfFFNN(ModelBase):
    def __init__(self, n_in, n_mid, n_out, batchsize, gpu=-1):
        # 基本設定
        self.n_in = n_in
        self.n_mid = n_mid
        self.n_out = n_out
        self.batchsize = batchsize
        self.gpu = gpu

        # モデル
        self.model = FFNN(n_in, n_mid, n_out)

        # GPU関連設定
        self.xp = np if gpu < 0 else cuda.cupy
        if gpu >= 0:
            cuda.get_device_from_id(gpu).use()
            cuda.check_cuda_available()
            self.model.to_gpu()

        # 最適化手法(Adam)
        self.optimizer = optimizers.Adam()
        self.optimizer.setup(self.model)
