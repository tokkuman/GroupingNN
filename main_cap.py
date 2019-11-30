# -*- coding: utf-8 -*-

import chainer
from chainer import serializers
import sys
import numpy as np
import cv2
import copy
from argparse import ArgumentParser

from model import Model
from model import Classifier

ap = ArgumentParser(description='python main_cap.py')
ap.add_argument('--model', '-m', nargs='?', default='0', help='Specify loading file path of learned Model')
ap.add_argument('--gpu', '-g', type=int, default=-1, help='Specify GPU ID (negative value indicates CPU)')
ap.add_argument('--frame', '-f', type=int, default=3, help='Specify Grouping frame rate (default = 3)')
ap.add_argument('--method', '-d', type=int, default=1, help='Specify Method Flag (1 : Haarcascades Frontalface Default, 2 : Haarcascades Frontalface Alt1, 3 : Haarcascades Frontalface Alt2, Without : Haarcascades Frontalface Alt Tree)')

args = ap.parse_args()

# GPU use flag
print 'GPU: {}'.format(args.gpu)

if __name__ == '__main__':

    group = ['Ravenclaw', 'Gryffindor', 'Hufflpuff', 'Slytherin']
    #group = ['Others', 'FunaLab', 'Others', 'FunaLab']

    model = Model()
    if not args.model == '0':
        try:
            serializers.load_hdf5(args.model, model)
            print 'Loading Model : ' + args.model
        except:
            print 'ERROR!!'
            print 'Usage : Input File Path of Model (ex ./hoge.model)'
            sys.exit()
    clf = Classifier(model)

    if args.method == 1:
        face_cascade = cv2.CascadeClassifier('haarcascade/haarcascade_frontalface_default.xml')
    elif args.method == 2:
        face_cascade = cv2.CascadeClassifier('haarcascade/haarcascade_frontalface_alt.xml')
    elif args.method == 3:
        face_cascade = cv2.CascadeClassifier('haarcascade/haarcascade_frontalface_alt2.xml')
    else:
        face_cascade = cv2.CascadeClassifier('haarcascade/haarcascade_frontalface_alt_tree.xml')
    cap = cv2.VideoCapture(0)
    cap.set(3,640)
    cap.set(4,480)
    cnt = 0
    while(True):
        ret, frame = cap.read()
        if ret == False:
            break
        else:
            if cnt % args.frame == 0:
                gImg = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                faces = face_cascade.detectMultiScale(gImg, 1.3, 5)
                if len(faces) > 0:
                    for num in range(len(faces)):
                        cropImg = copy.deepcopy(frame[faces[num][1]:faces[num][1]+faces[num][3], faces[num][0]:faces[num][0]+faces[num][2]])
                        resizeImg = cv2.resize(cropImg, (100, 100))
                        resizeImg = resizeImg.astype(np.float32) / resizeImg.max()
                        testImg = copy.deepcopy(resizeImg.reshape(1, 100, 100, 3).transpose(0, 3, 1, 2))
                        pre = clf.predictor(testImg, train=False)

                        text = group[np.argmax(pre.data)]
                        x, y, w, h = faces[num]
                        if np.argmax(pre.data) == 0:  # Re
                            color = (255, 0, 0)
                        elif np.argmax(pre.data) == 1:  # Gr
                            color = (0, 0, 255)
                        elif np.argmax(pre.data) == 2:  # Hu
                            color = (0, 255, 255)
                        elif np.argmax(pre.data) == 3:  # Sl
                            color = (0, 255, 0)
                        else:
                            color = (0, 0, 0)
                        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                        cv2.putText(frame, text, (x+(w/5), y+h+30), cv2.FONT_HERSHEY_COMPLEX, 1.0, color, thickness=3)
                cv2.imshow('fram', frame)
            cnt += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
