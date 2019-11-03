# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 12:56:31 2019

@author: dell
"""

import skimage.io as io
from glob import glob
VOC_path = 'D://dataset/voc/VOCdevkit/VOC2012/JPEGImages'
Train_path = 'F://university/course/4th_1/acquisition/prj/train'
Eval_path = 'F://university/course/4th_1/acquisition/prj/eval/Pair'
VOC_names = glob(VOC_path+'\*')
'''
for i in range(2000):
    img = io.imread(VOC_names[i],as_grey=True)
    io.imsave(Train_path+'/train_%d.jpg'%i, img)
    print('%d finished' % i)
'''
for i in range(2000,2020):
    img = io.imread(VOC_names[i], as_grey=True)
    io.imsave(Eval_path + '/eval_%d.jpg' % i, img)
    print('%d finished' % i)