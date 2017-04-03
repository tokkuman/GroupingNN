# -*- coding: utf-8 -*-

import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions
from chainer.links import caffe
from chainer import link

import sys
import time
import random
import copy
import math
import six
import os
import os.path as pt
import numpy as np
import cv2
import matplotlib.pyplot as plt
import skimage.io as skimage
from argparse import ArgumentParser

from model import Model
from model import Classifier

plt.style.use('ggplot')

ap = ArgumentParser(description='python main.py')
ap.add_argument('--indir', '-i', nargs='?', default='unko.jpg', help='Specify input files directory training data')
ap.add_argument('--outdir', '-o', nargs='?', default='Result', help='Specify output files directory for create result and save model file')
ap.add_argument('--model', '-m', nargs='?', default='0', help='Specify loading file path of learned Model')
ap.add_argument('--gpu', '-g', type=int, default=-1, help='Specify GPU ID (negative value indicates CPU)')
ap.add_argument('--imsize', '-s', type=int, default=100, help='Specify size of image')
ap.add_argument('--kcross', '-k', type=int, default=5, help='Specify cross validation of k (0 : Training Only)')
ap.add_argument('--epoch', '-e', type=int, default=10, help='Specify number of sweeps over the dataset to train')
ap.add_argument('--batchsize', '-b', type=int, default=1, help='Specify batchsize')
ap.add_argument('--phase', '-p', type=int, default=1, help='Specify Phase (0 : training Phase, 1 : prediction Phase)')
ap.add_argument('--method', '-d', type=int, default=1, help='Specify Method Flag (1 : Haarcascades Frontalface Default, 2 : Haarcascades Frontalface Alt1, 3 : Haarcascades Frontalface Alt2, Without : Haarcascades Frontalface Alt Tree)')

args = ap.parse_args()
opbase = args.outdir
argvs = sys.argv

# GPU use flag
print 'GPU: {}'.format(args.gpu)
# Path Separator
psep = '/'
if (args.outdir[len(opbase) - 1] == psep):
    opbase = opbase[:len(opbase) - 1]
if not (args.outdir[0] == psep):
    if (args.outdir.find('./') == -1):
        opbase = './' + opbase
# Create Opbase
t = time.ctime().split(' ')
if t.count('') == 1:
    t.pop(t.index(''))
opbase = opbase + '_' + t[1] + t[2] + t[0] + '_' + t[4] + '_' + t[3].split(':')[0] + t[3].split(':')[1] + t[3].split(':')[2]
if not (pt.exists(opbase)):
    os.mkdir(opbase)
    print 'Output Directory not exist! Create...'
print 'Output Directory:', opbase

def loadImages(path):
    imagePathes = map(lambda a:os.path.join(path,a),os.listdir(path))
    images = np.array(map(lambda x: skimage.imread(x), imagePathes))
    return images

def TrainingFhase(trainData, trainLabel, testLabel, clf, opt):
    bestTrainAcc, bestEvalAcc = 0, 0
    trainAccList, evalAccList = [], []
    for epoch in range(args.epoch):
        ### Training
        trainSumLoss, trainSumAcc = 0, 0
        trainPerm = np.random.permutation(trainLabel)
        for num in range(len(trainLabel)):
            x, y = trainData[group[trainPerm[num][0]]][trainPerm[num][1]].reshape(1, 100, 100, 3).transpose(0, 3, 1, 2), np.array([trainPerm[num][0]]).astype(np.int32)
            x, y = Variable(x), Variable(y)
            if args.gpu == 0:
                x.to_gpu()
                y.to_gpu()
            loss = clf(x, y, train=True)
            opt.zero_grads()
            loss.backward()    # back propagation
            opt.update()       # update parameters
            trainSumAcc += clf.acc.data * args.batchsize
            trainSumLoss += clf.loss.data * args.batchsize
        trainAcc = trainSumAcc / len(trainPerm)
        trainLoss = trainSumLoss / len(trainPerm)

        ### Evaluation
        evalSumLoss,  evalSumAcc = 0, 0
        testPerm = np.random.permutation(testLabel)
        for num in range(len(testLabel)):
            x, y = trainData[group[testPerm[num][0]]][testPerm[num][1]].reshape(1, 100, 100, 3).transpose(0, 3, 1, 2), np.array([testPerm[num][0]]).astype(np.int32)
            x, y = Variable(x), Variable(y)
            if args.gpu == 0:
                x.to_gpu()
                y.to_gpu()
            loss = clf(x, y, train=False)
            evalSumAcc += clf.acc.data * args.batchsize
            evalSumLoss += clf.loss.data * args.batchsize
        evalAcc = evalSumAcc / len(testPerm)
        evalLoss = evalSumLoss / len(testPerm)

        print '===================================='
        print 'epoch :', epoch+1
        print 'TrainLoss :', trainLoss, ', TrainAccuracy :', trainAcc
        print 'TestLoss :', evalLoss, ', TestAccuracy :', evalAcc

        trainAccList.append(1 - trainAcc)
        evalAccList.append(1 - evalAcc)

        filename = opbase + psep + 'result.txt'
        f = open(filename, 'a')
        f.write('==================================\n')
        f.write('epoch : '.format(str(epoch+1)) + '\n')
        f.write('TrainLoss={}, TrainAccuracy={}'.format(trainLoss, trainAcc) + '\n')
        f.write('TestLoss={}, TestAccuracy={}'.format(evalLoss, evalAcc) + '\n')
        f.close()

        # Save Model
        if bestTrainAcc < trainAcc:
            modelfile = 'ModelBThdf5.model'
            serializers.save_hdf5(opbase + psep + modelfile, model)
        if bestEvalAcc < evalAcc:
            modelfile = 'ModelBEhdf5.model'
            serializers.save_hdf5(opbase + psep + modelfile, model)

    filename = opbase + psep + 'result.txt'
    f = open(filename, 'a')
    f.write('==================================\n')
    f.write('BestTrainAccuracy={}'.format(bestTrainAcc) + '\n')
    f.write('BestEvaluationAccuracy={}'.format(bestEvalAcc) + '\n')
    f.close()
    
    return trainAccList, evalAccList


if __name__ == '__main__':

    group = ['Gryffindor', 'Ravenclaw', 'Hufflpuff', 'Slytherin']
    
    if args.phase == 0:  ## training phase
        trainImg = {}
        trainLabel, testLabel = [], []
        intmax = 0
        for g in group:
            trainImg[g] = loadImages(args.indir + '/' + g)
            perm = np.random.permutation(len(trainImg[g]))
            for i in range(len(trainImg[g])):
                if args.kcross == 0:
                    trainLabel.append([group.index(g), i])
                    testLabel.append([group.index(g), i])
                else:
                    if len(trainImg[g]) * (args.kcross - 1) / args.kcross > i:
                        trainLabel.append([group.index(g), perm[i]])
                    else:
                        testLabel.append([group.index(g), perm[i]])
            intmax = np.max([intmax, trainImg[g].max()])
        trainLabel = np.array(trainLabel).astype(np.int32)
        testLabel = np.array(testLabel).astype(np.int32)
        for g in group:
            trainImg[g] = trainImg[g].astype(np.float32)
            trainImg[g] = trainImg[g] / intmax

        model = Model()
        if not args.model == '0':
            try:
                serializers.load_hdf5(args.model, model)
                print 'Loading Model : ' + args.model
                filename = opbase + psep + 'result.txt'
                f = open(filename, 'w')
                f.write('Loading Model : {}\n'.format(args.model))
                f.close()
            except:
                print 'ERROR!!'
                print 'Usage : Input File Path of Model (ex ./hoge.model)'
                sys.exit()
        if args.gpu == 0:
            cuda.get_device(args.gpu).use()  # Make a specified GPU current
            model.to_gpu()
        clf = Classifier(model)
        opt = optimizers.Adam()
        opt.setup(clf)
        opt.add_hook(chainer.optimizer.WeightDecay(0.0001))

        filename = opbase + psep + 'result.txt'
        f = open(filename, 'w')
        f.write('python ' + ' '.join(argvs) + '\n')
        f.write('[Hyperparameter of Learning Properties]\n')
        f.write('Output Directory : {}\n'.format(opbase))
        f.write('GPU: {}\n'.format(args.gpu))
        f.write('number of Training Data : {}\n'.format(len(trainLabel)))
        f.write('number of Training Data : {}\n'.format(len(testLabel)))
        f.close()
    
        trainAccList, evalAccList = TrainingFhase(trainImg, trainLabel, testLabel, clf, opt)    # Training & Validatioin
        plt.figure(figsize=(8,6))
        plt.plot(range(1, args.epoch + 1, 1), trainAccList)
        plt.plot(range(1, args.epoch + 1, 1), evalAccList)
        #plt.ylim(0.0, np.max([np.max(trainAccList), np.max(evalAccList)]) + 0.2)
        plt.legend(["train_error", "test_error"],loc=1) # upper right
        plt.title("Error Rate of Train and Test")
        plt.xlabel('Epoch')
        plt.ylabel('Error Rate')
        plt.plot()
        figname = 'CNN_ErrorRate_epoch' + str(args.epoch) + '.pdf'
        plt.savefig(opbase + '/' + figname)


    if args.phase == 1:  ## prediction phase
        img = skimage.imread(args.indir)
        model = Model()
        if not args.model == '0':
            try:
                serializers.load_hdf5(args.model, model)
                print 'Loading Model : ' + args.model
                filename = opbase + psep + 'result.txt'
                f = open(filename, 'w')
                f.write('Loading Model : {}\n'.format(args.model))
                f.close()
            except:
                print 'ERROR!!'
                print 'Usage : Input File Path of Model (ex ./hoge.model)'
                sys.exit()
        clf = Classifier(model)

        preImg, text = [], []
        gImg = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        if args.method == 1:
            face_cascade = cv2.CascadeClassifier('haarcascade/haarcascade_frontalface_default.xml')
        elif args.method == 2:
            face_cascade = cv2.CascadeClassifier('haarcascade/haarcascade_frontalface_alt.xml')
        elif args.method == 3:
            face_cascade = cv2.CascadeClassifier('haarcascade/haarcascade_frontalface_alt2.xml')
        else:
            face_cascade = cv2.CascadeClassifier('haarcascade/haarcascade_frontalface_alt_tree.xml')
        faces = face_cascade.detectMultiScale(gImg, 1.3, 5)
        for num in range(len(faces)):
            cropImg = copy.deepcopy(img[faces[num][1]:faces[num][1]+faces[num][3], faces[num][0]:faces[num][0]+faces[num][2]])
            resizeImg = cv2.resize(cropImg, (args.imsize, args.imsize))
            resizeImg = resizeImg.astype(np.float32) / resizeImg.max()
            testImg = copy.deepcopy(resizeImg.reshape(1, args.imsize, args.imsize, 3).transpose(0, 3, 1, 2))
            pre = clf.predictor(testImg, train=False)

            print group[np.argmax(pre.data)]
            text = group[np.argmax(pre.data)]
            x, y, w, h = faces[num]
            if np.argmax(pre.data) == 0:
                color = (255, 0, 0)
            elif np.argmax(pre.data) == 1:
                color = (0, 0, 255)
            elif np.argmax(pre.data) == 2:
                color = (255, 255, 0)
            elif np.argmax(pre.data) == 3:
                color = (0, 255, 0)
            else:
                color = (0, 0, 0)
            cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
            cv2.putText(img, text, (x+(w/5), y+h+30), cv2.FONT_HERSHEY_COMPLEX, 1.0, color, thickness=3)

        filename = opbase + psep + 'result.jpg'
        cv2.imwrite(filename, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
