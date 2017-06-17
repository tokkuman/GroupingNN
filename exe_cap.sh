#!/bin/sh

if [ ! -e LearnedModel.model ]; then
    wget https://fun.bio.keio.ac.jp/~tokuoka/Downloads/LearnedModel.model -P .
fi
python main_cap.py -m LearnedModel.model