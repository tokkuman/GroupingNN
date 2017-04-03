# -*- coding: utf-8 -*-

import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions

class Model(Chain):
    def __init__(self):
        super(Model, self).__init__(
            conv1=L.Convolution2D(3, 128, 7, stride=1),
            bn2=L.BatchNormalization(128),
            conv3=L.Convolution2D(128, 256, 5, stride=1),
            bn4=L.BatchNormalization(256),
            conv5=L.Convolution2D(256, 384, 3, stride=1),
            bn6=L.BatchNormalization(384),
            fc7=L.Linear(6144, 8192),
            fc8=L.Linear(8192, 1024),
            fc9=L.Linear(1024, 4),
        )

    def __call__(self, x, train=True):
        h = F.max_pooling_2d(self.bn2(F.relu(self.conv1(x))), 3, stride=3)
        h = F.max_pooling_2d(self.bn4(F.relu(self.conv3(h))), 3, stride=3)
        h = F.max_pooling_2d(self.bn6(F.relu(self.conv5(h))), 2, stride=2)
        h = F.dropout(F.relu(self.fc7(h)), train=train)
        h = F.dropout(F.relu(self.fc8(h)), train=train)
        y = self.fc9(h)
        return y

class Classifier(Chain):
    def __init__(self, predictor):
        super(Classifier, self).__init__(predictor=predictor)
        self.train = True

    def __call__(self, x, t, train=True):
        y = self.predictor(x, train)
        self.loss = F.softmax_cross_entropy(y, t)
        self.acc = F.accuracy(y, t)
        return self.loss
