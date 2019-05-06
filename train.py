#!/usr/bin/evn python
# -*- coding: utf-8 -*-
# Copyright (c) 2018 - xibin.yue <xibin.yue@moji.com>
from model.nnModel import *
from model.corpusSolver import *
import torch

dataClass = Corpus('./corpus/qingyun.tsv', maxSentenceWordsNum=25)


model = Seq2Seq(dataClass, featureSize=256, hiddenSize=256,
                attnType='L', attnMethod='general',
                encoderNumLayers=3, decoderNumLayers=2,
                encoderBidirectional=True,
                device=torch.device('cuda:0'))
model.train(batchSize=1024, epoch=500)
model.save('model.pkl')
