# --------------------------------------------------------
# TRIPLET LOSS
# Copyright (c) 2015 Pinguo Tech.
# Written by David Lu
# --------------------------------------------------------

"""The data layer used during training a VGG_FACE network by triplet loss.
   The layer combines the input image into triplet.Priority select the semi-hard samples
"""
import caffe
import numpy as np
from numpy import *
import yaml
from multiprocessing import Process, Queue
from caffe._caffe import RawBlobVec
from sklearn import preprocessing
import math
import config

class TripletSelectLayer(caffe.Layer):
        
    def setup(self, bottom, top):
        """Setup the TripletSelectLayer."""
        self.triplet = config.BATCH_SIZE/3
        top[0].reshape(self.triplet,shape(bottom[0].data)[1])
        top[1].reshape(self.triplet,shape(bottom[0].data)[1])
        top[2].reshape(self.triplet,shape(bottom[0].data)[1])

    def forward(self, bottom, top):
        """Get blobs and copy them into this layer's top blob vector."""
        top_archor = []
        top_positive = []
        top_negative = []
        labels = []
        self.tripletlist = []
        self.no_residual_list=[]
        aps = {}
        ans = {}

        archor_feature = bottom[0].data[0]
        for i in range(self.triplet):
            positive_feature = bottom[0].data[i+self.triplet]
            a_p = archor_feature - positive_feature
            ap = np.dot(a_p,a_p)
            aps[i+self.triplet] = ap
        aps = sorted(aps.items(), key = lambda d: d[1], reverse = True)
        for i in range(self.triplet):
            negative_feature = bottom[0].data[i+self.triplet*2]
            a_n = archor_feature - negative_feature
            an = np.dot(a_n,a_n)
            ans[i+self.triplet*2] = an
        ans = sorted(ans.items(), key = lambda d: d[1], reverse = False) # We want the lowest distance to be first

        for i in range(self.triplet): 
            top_archor.append(bottom[0].data[i])
            top_positive.append(bottom[0].data[aps[i][0]])
            top_negative.append(bottom[0].data[ans[i][0]])
            if aps[i][1] >= ans[i][1]:
               self.no_residual_list.append(i)
            self.tripletlist.append([i,aps[i][0],ans[i][0]])

        top[0].data[...] = np.array(top_archor).astype(float32)
        top[1].data[...] = np.array(top_positive).astype(float32)
        top[2].data[...] = np.array(top_negative).astype(float32)
    

    def backward(self, top, propagate_down, bottom):
        
        for i in range(len(self.tripletlist)):
            if not i in self.no_residual_list:
                bottom[0].diff[self.tripletlist[i][0]] += top[0].diff[i]
                bottom[0].diff[self.tripletlist[i][1]] += top[1].diff[i]
                bottom[0].diff[self.tripletlist[i][2]] += top[2].diff[i]
            else:
                bottom[0].diff[self.tripletlist[i][0]] += np.zeros(shape(top[0].diff[i]))
                bottom[0].diff[self.tripletlist[i][1]] += np.zeros(shape(top[1].diff[i]))
                bottom[0].diff[self.tripletlist[i][2]] += np.zeros(shape(top[2].diff[i]))

        #print 'backward-no_re:',bottom[0].diff[0][0]
        #print 'tripletlist:',self.no_residual_list

        

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass







