#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 09:54:43 2019

@author: hu
"""
import os
import sys
import numpy as np
from Trajectory import Trajectory, Node, IOU, KalmanFilter, chi2inv95, TrackState
import cv2
import lap
import itertools
import random
import tensorflow as tf
import keras
import keras.layers as KL
import keras.models as KM
import keras.backend as K
from lib.utils.shell import MODELS
parpath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
curpath = os.path.dirname(os.path.abspath(__file__))
sys.path.append(parpath)
from lib.tools.config import Config
from lib.utils.shell import MODELS
from lib.tools.DataAugmentation import DataAugmentation as DA
from lib.dla_34 import DLASeg
from lib.tools.load_weights import load_weights_by_name
import math
def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m+1,-n:n+1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h
def draw_umich_gaussian(heatmap, center, radius, intense, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)
    x, y = int(center[0]), int(center[1])
    height, width = heatmap.shape[0:2]
    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)
    masked_heatmap  = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0: # TODO debug
      np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    if x > 0 and x < 272 and y > 0 and y < 152:
        heatmap[y, x] = intense
    return heatmap

def gaussian_radius(det_size, min_overlap=0.7):
    height, width = det_size
    a1  = 1
    b1  = (height + width)
    c1  = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1  = (b1 + sq1) / 2
    a2  = 4
    b2  = 2 * (height + width)
    c2  = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2  = (b2 + sq2) / 2
    a3  = 4 * min_overlap
    b3  = -2 * min_overlap * (height + width)
    c3  = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3  = (b3 + sq3) / 2
    return min(r1, r2, r3)

class Trajectories():
    def __init__(self, seqInfo, config = None, CFG = None):
        self.seqInfo = seqInfo
        self.config = config
        self.DA = DA('validation', self.config)
        #New, Tracked, Lost, Invalid, Removed tracks
        self.New = []
        self.Tracked = []
        self.Lost = []
        self.Invalid = []
        self.Removed = []
        
        self.results = []
        
        self.max_time_lost = 30
        
        self.graph()
        self.sess = K.get_session()
        self.sess.run(K.update(self.model.get_layer('dlaseg.free.conv1').weights[0], self.model.get_layer('dlaseg.hm.conv1').weights[0]))
        self.sess.run(K.update(self.model.get_layer('dlaseg.free.conv1').weights[1], self.model.get_layer('dlaseg.hm.conv1').weights[1]))
        self.sess.run(K.update(self.model.get_layer('dlaseg.free.conv2').weights[0], self.model.get_layer('dlaseg.hm.conv2').weights[0]))
        self.sess.run(K.update(self.model.get_layer('dlaseg.free.conv2').weights[1], self.model.get_layer('dlaseg.hm.conv2').weights[1]))
    
    def l1_loss(self, a, b):
        return tf.reduce_mean(tf.abs(a - b))
    
    def FocalLoss(self, phm, ghm):
        phm = tf.clip_by_value(phm, K.epsilon(), 1)
        positive_mask = tf.where(tf.greater_equal(ghm, 1), ghm, tf.zeros_like(ghm))
        negative_mask = tf.where(tf.logical_and(tf.less(ghm, 1), tf.greater(ghm, 0)), tf.ones_like(ghm), tf.zeros_like(ghm))
        negative_weights = tf.pow(1-ghm, 4)
        positive_loss = tf.log(phm) * tf.pow(1-phm, 2) * positive_mask
        negative_loss = tf.log(1-phm) * tf.pow(phm, 2) * negative_mask * negative_weights
        num_positive = tf.clip_by_value(tf.reduce_sum(positive_mask), 1, 1e10)
        positive_loss = tf.reduce_sum(positive_loss)
        negative_loss = tf.reduce_sum(negative_loss)
        return -(positive_loss + negative_loss)/num_positive
    
    def decisionMatrix(self,):
        probes_t = tf.matmul(self.tf_probes, self.kernel) + self.bias
        gallery_t = tf.matmul(self.tf_gallery, self.kernel) + self.bias
        probes_t = tf.nn.l2_normalize(probes_t, axis = -1)
        gallery_t = tf.nn.l2_normalize(gallery_t, axis = -1)
        self.sim = tf.reduce_sum(tf.expand_dims(probes_t, axis = 1) * \
                                 tf.expand_dims(gallery_t, axis = 0), axis = -1)
        
        self.h_sim, self.v_sim = self.softmatrix(self.sim)
        self.o_sim = tf.reduce_sum(tf.expand_dims(tf.nn.l2_normalize(self.tf_probes, axis = -1), axis = 1) * \
                                 tf.expand_dims(tf.nn.l2_normalize(self.tf_gallery, axis = -1), axis = 0), axis = -1)
            
    def decisionLoss(self,):
        h_positive = tf.gather_nd(self.h_sim, self.tf_matches)
        v_positive = tf.gather_nd(self.v_sim, self.tf_matches)
        sim = tf.tensor_scatter_nd_update(self.sim, self.tf_matches, tf.cast(tf.ones_like(self.tf_matches[:, 0])*-1, 'float32'))
        h_sim, v_sim = self.softmatrix(sim)
        h_negative = tf.gather(h_sim[:, -1], self.tf_matches[:, 0])
        v_negative = tf.gather(v_sim[-1, :], self.tf_matches[:, 1])
        loss = -(tf.reduce_sum(tf.log(h_positive))+\
                 tf.reduce_sum(tf.log(v_positive))+\
                 tf.reduce_sum(tf.log(h_negative))+\
                 tf.reduce_sum(tf.log(v_negative)))/\
                 tf.cast(tf.shape(self.tf_matches)[0]*4, 'float32')
        return loss
    
    def backprop(self, grads, params, name, lr=0.01):
        if not hasattr(self, name+'moments'):
            setattr(self, name+'shapes', [K.int_shape(p) for p in params])
            setattr(self, name + 'moments', [K.zeros(shape) for shape in eval('self.'+name+'shapes')])
        moments = eval('self.'+name+'moments')
        updates = []
        for p, g, m in zip(params, grads, moments):
            v = self.momentum * m - lr * g  # velocity
            updates.append(K.update(m, v))
            if self.nesterov:
                new_p = p + self.momentum * v - lr * g
            else:
                new_p = p + v
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)
            updates.append(K.update(p, new_p))
        setattr(self, name+'updates', updates)
           
    
    def graph(self,):
        #training parameters
        self.lr = 0.2
        self.pad = 0.5
        self.nesterov = False
        self.decay = K.variable(0, name='decay')
        self.momentum = K.variable(0.9, name='momentum')
        #scene-adaptive detection
        #feedback
        self.tf_matches = tf.placeholder('int32', name = 'matches')
        self.tf_whs = tf.placeholder('float32', name = 'whs')
        self.tf_regs = tf.placeholder('float32', name = 'regs')
        self.tf_hm = tf.placeholder('float32', name = 'hm')
        #model
        self.model = DLASeg(self.config).model_pure()
        load_weights_by_name(self.model, 'dla_34_new_reid.h5')
        #detection
        self.input_image = self.model.get_layer('input_image').input
        self.det_boxes = self.model.get_layer('detection').output[0]
        self.det_scores = self.model.get_layer('detection').output[1]
        self.det_whs = self.model.get_layer('detection').output[2]
        self.det_regs = self.model.get_layer('detection').output[3]
        #re-id
        self.det_reid = self.model.get_layer('reid').output[0]
        #parameters--network weights
        self.hm = self.model.get_layer('dlaseg.hm.conv2').output
        self.free = self.model.get_layer('dlaseg.free.conv2').output
        #self.hm_params = [#self.model.get_layer('dlaseg.free.conv0').weights[0], \
                          #self.model.get_layer('dlaseg.free.conv0').weights[1], \
                          #self.model.get_layer('dlaseg.free.conv1').weights[0], \
                          #self.model.get_layer('dlaseg.free.conv1').weights[1], \
        #                  self.model.get_layer('dlaseg.free.conv2').weights[0], \
        #                  self.model.get_layer('dlaseg.free.conv2').weights[1]
        #                  ]
        self.hm_params = [self.model.get_layer('dlaseg.hm.conv2').weights[0], \
                          self.model.get_layer('dlaseg.hm.conv2').weights[1]]
        self.wh = self.model.get_layer('dlaseg.wh.conv2').output
        self.wh_params = [self.model.get_layer('dlaseg.wh.conv2').weights[0], \
                          self.model.get_layer('dlaseg.wh.conv2').weights[1]]
        self.reg = self.model.get_layer('dlaseg.reg.conv2').output
        self.reg_params = [self.model.get_layer('dlaseg.reg.conv2').weights[0], \
                           self.model.get_layer('dlaseg.reg.conv2').weights[1]]
        #losses
        self.hm_loss = self.FocalLoss(tf.nn.sigmoid(self.hm), self.tf_hm)
        self.reg_loss = self.l1_loss(self.tf_regs, tf.gather(self.det_regs[0], self.tf_matches[:, 1]))
        self.wh_loss = self.l1_loss(self.tf_whs, tf.gather(self.det_whs[0], self.tf_matches[:, 1]))
        
        #gradients
        hm_grads = K.gradients(self.hm_loss, self.hm_params)
        wh_grads = K.gradients(self.wh_loss, self.wh_params)
        reg_grads = K.gradients(self.reg_loss, self.reg_params)
        
        #back propogation
        self.backprop(hm_grads, self.hm_params, name = 'hm', lr = 0.001)
        self.backprop(wh_grads, self.wh_params, name = 'wh', lr = 0.000)
        self.backprop(reg_grads, self.reg_params, name = 'reg', lr = 0.00)
        
        #scene-adaptive transformer
        #feedback
        self.tempa = tf.Variable(1, dtype = 'float32', trainable = True)
        self.tf_probes = tf.placeholder('float32', name = 'probes')
        self.tf_gallery = tf.placeholder('float32', name = 'gallery')
        #paramters--network weights
        self.kernel = tf.Variable(np.eye(1536), dtype = 'float32', trainable = True)
        self.bias = tf.Variable(np.zeros((1536)), dtype = 'float32', trainable = True)
        self.tempa = tf.Variable(1, dtype = 'float32', trainable = True)
        self.params = [self.kernel, self.bias, self.tempa]
        
        #decision matrices
        self.decisionMatrix()
        #loss
        loss = self.decisionLoss()
        #gradients
        grads = K.gradients(loss, self.params)
        #back propogation
        self.backprop(grads, self.params, name='',lr = self.lr)
        
        
        #tools
        #get dis
        self.tf_boxes1 = tf.placeholder('float32', name = 'boxes1')
        self.tf_boxes2 = tf.placeholder('float32', name = 'boxes2')
        boxes1 = tf.expand_dims(self.tf_boxes1, axis = 1)
        boxes2 = tf.expand_dims(self.tf_boxes2, axis = 0)
        d1 = tf.exp(-(tf.pow(boxes1[...,0]-boxes2[...,0], 2)/(2*tf.pow(boxes1[...,2]/2, 2))+\
                      tf.pow(boxes1[...,1]-boxes2[...,1], 2)/(2*tf.pow(boxes1[...,3]/2, 2))))
            
        d2 = tf.exp(-(tf.pow(boxes1[...,0]-boxes2[...,0], 2)/(2*tf.pow(boxes2[...,2]/2, 2))+\
                      tf.pow(boxes1[...,1]-boxes2[...,1], 2)/(2*tf.pow(boxes2[...,3]/2, 2))))
            
        d3 = tf.exp(-(tf.pow(boxes1[...,3]-boxes2[...,3], 2)/\
                      (2*(tf.pow(boxes1[...,3]/2,2)+tf.pow(boxes2[...,3]/2,2)))))
        
        self.dis = tf.pow(d1*d2*d3,1/3.)
        
        #get sigmoid
        self.tf_sigmoid = tf.placeholder('float32', name = 'sigmoid')
        self.sigmoid = tf.nn.sigmoid(self.tf_sigmoid)
        
        #get softmax
        self.tf_softmax = tf.placeholder('float32', name = 'softmax')
        self.h_softmax, self.v_softmax = self.softmatrix(self.tf_softmax)
    
    
    def getdis(self, boxes1, boxes2):
        if len(boxes1) == 0 or len(boxes2) == 0:
            return np.zeros((len(boxes1), len(boxes2)))
        return self.sess.run(self.dis, {self.tf_boxes1:boxes1, self.tf_boxes2:boxes2})
    
    def getsim(self, probes, gallery):
        if len(probes) == 0 or len(gallery) == 0:
            return np.zeros((len(probes), len(gallery)))
        return self.sess.run(self.sim, {self.tf_probes:probes, self.tf_gallery:gallery})
    
    def getsigmoid(self, inputs):
        return self.sess.run(self.sigmoid, {self.tf_sigmoid:inputs})
    
    def getsoftmax(self, inputs):
        return self.sess.run([self.h_softmax, self.v_softmax], {self.tf_softmax:inputs})
        
    def forward_detection(self, image):
        image, _, _, self.meta = self.DA(image, [])
        self.detection_handle = \
            self.sess.partial_run_setup([self.det_boxes,\
                                         self.det_scores,\
                                         self.det_reid,\
                                         self.hm, \
                                         self.free,\
                                         self.det_whs,\
                                         self.hmupdates,\
                                         self.whupdates,\
                                         self.regupdates,\
                                         ],\
                                        [self.input_image, self.tf_hm, self.tf_whs, self.tf_regs, self.tf_matches])
        
        return self.sess.partial_run(self.detection_handle, \
                                    [self.det_boxes, \
                                     self.det_scores, \
                                     self.det_reid,\
                                     self.hm,\
                                     self.free,\
                                     ],\
                                    {self.input_image:np.array([image])})
    
    def backward_detection(self, matches, scores):
        if len(matches) == 0:return
        hm, wh, reg = self.create_heatmap(matches, scores)
        self.sess.partial_run(self.detection_handle, \
                             [self.hmupdates], \
                             {self.tf_hm:np.expand_dims(hm, axis = -1),\
                              self.tf_whs:np.array([wh]), \
                              self.tf_regs:np.array([reg]),\
                              self.tf_matches:np.array(matches)})
        
    def create_heatmap(self, matches, scores):
        oh, ow = 608//4, 1088//4
        hm = np.zeros((1, oh, ow), dtype=np.float32)
        wh = np.zeros((len(matches), 2), dtype = np.float32)
        reg = np.zeros((len(matches), 2), dtype = np.float32)
        boxes, ss = [], []
        tracks = self.Tracked + self.Lost
        
        for trackid, detid in matches:
            ss.append(scores[detid])
            track = tracks[trackid]
            if track.max_score < 0.7:continue
            cx, cy, a, h = track.kf.mean[:4]
            w = a * h
            x, y = cx - 0.5 * w, cy - 0.5 * h
            boxes.append([x, y, x+w, y+h])
        if len(boxes) == 0:
            return hm, wh, reg
        boxes = np.array(boxes)
        scale_x = scale_y = self.meta[0][11]
        window = self.meta[0][7:11]
        boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale_x + window[1]
        boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale_y + window[0]
        draw_gaussian = draw_umich_gaussian
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box/4
            cy, cx, h, w = (y1+y2)/2., (x1+x2)/2., y2-y1, x2-x1
            intense = 1
            if h > 0 and w > 0:
                radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                radius = max(0, int(radius))
                ct = np.array([cx, cy], dtype=np.float32)
                ct_int = ct.astype(np.int32)
                draw_gaussian(hm[0], ct_int, radius, intense)
                wh[i] = 1. * w, 1. * h
                reg[i] = ct - ct_int
        return hm, wh, reg
        
    def softmatrix(self, sim):
        h_sim = tf.concat([sim, tf.ones_like(sim[:, :1])*self.pad], axis = -1)
        v_sim = tf.concat([sim, tf.ones_like(sim[:1, :])*self.pad], axis = 0)
        h_sim = tf.nn.softmax(h_sim * self.tempa, axis = -1)
        v_sim = tf.transpose(tf.nn.softmax(tf.transpose(v_sim, [1, 0]) * self.tempa, axis = -1), [1, 0])
        return h_sim, v_sim
        
    def forward(self, probes, gallery):
        if len(probes) == 0 or len(gallery) == 0:
            return np.zeros((len(probes), len(gallery))), np.zeros((len(probes), len(gallery))), np.zeros((len(probes), len(gallery)+1)), np.zeros((len(probes)+1, len(gallery)))
        self.handle = self.sess.partial_run_setup([self.sim, self.o_sim, self.h_sim, self.v_sim, self.updates], [self.tf_probes, self.tf_gallery, self.tf_matches])
        return self.sess.partial_run(self.handle, [self.sim, self.o_sim, self.h_sim, self.v_sim], {self.tf_probes:np.array(probes), self.tf_gallery:np.array(gallery)})
    
    #def backward(self,hmatches, vmatches):
    def backward(self, matches):
        if len(matches) == 0:return
        self.sess.partial_run(self.handle, self.updates[::-1], feed_dict = {self.tf_matches:np.array(matches)})
        
    
    def predict(self,):
        self.prediction = np.array([t.predict() for t in self.Tracked + self.Lost + self.New])
        return self.prediction
    
    def drawTrajectories(self, img, frame_id):
        for i, t in enumerate(self.Tracked + self.New):
            if t.id == -1:
                cx, cy, a, h = t.kf.mean[:4]
                w = a * h
                x, y = cx - 0.5 * w, cy - 0.5 * h
                img = cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), (255, 255, 255), 1)
                img = cv2.putText(img, '%d|%d'%(t.id, i), (int(x), int(y - 12)), cv2.FONT_HERSHEY_TRIPLEX , 0.5, 
                                        (255, 255, 255), 1)
                continue
            cx, cy, a, h = t.kf.mean[:4]
            w = a * h
            x, y = cx - 0.5 * w, cy - 0.5 * h
            img = cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), (int(t.c[0]), int(t.c[1]), int(t.c[2])), 2)
            img = cv2.putText(img, '%d|%d'%(t.id, i), (int(x), int(y - 12)), cv2.FONT_HERSHEY_TRIPLEX , 0.5, 
                                    (int(t.c[0]), int(t.c[1]), int(t.c[2])), 1)
        for i, t in enumerate(self.Lost):
            if frame_id - t.frame_id <= 2:
                cx, cy, a, h = t.kf.mean[:4]
                w = a * h
                x, y = cx - 0.5 * w, cy - 0.5 * h
                img = cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), (int(t.c[0]), int(t.c[1]), int(t.c[2])), 1)
                img = cv2.putText(img, '%d|%d|lost'%(t.id, i), (int(x), int(y - 12)), cv2.FONT_HERSHEY_TRIPLEX , 0.5, 
                                        (int(t.c[0]), int(t.c[1]), int(t.c[2])), 1)
                
        cv2.imwrite('results/%06d.jpg'%(frame_id + 1), img)
    
    def appendResult(self, frame_id):
        for t in self.Tracked + self.New + self.Lost:
            if t.id == -1:continue
            if (t.state == TrackState.Lost) and (frame_id - t.frame_id) > 2:continue
            cx, cy, a, h = t.kf.mean[:4]
            w = a * h
            x, y = cx - 0.5 * w, cy - 0.5 * h
            self.results.append([frame_id + 1, t.id + 1, x, y, w, h])
    
    def outputResults(self, name):
        f_result = open(name, 'w')
        for result in self.results:
            line = '%d, %d, %.3f, %.3f, %.3f, %.3f, 1, -1, -1, -1\n'%tuple(result)
            f_result.write(line)
        f_result.close()
    
    def createNode(self, frame_id, det,):
        feature, box, score = det[5:], det[:4], det[4]
        association_dict = {'box' : box,
                            'score' : score,
                            'feature':feature,
                            'occlusion':False,
                            'force' : False}
        return Node(frame_id = frame_id, association_dict = association_dict)
    
    @staticmethod
    def xywh2ccwh(boxes):
        boxes = boxes.copy()
        boxes[:, :2] += (boxes[:, 2:4] * 0.5)
        return boxes
    
    def embedding_match(self, em, hsm, vsm, distance, thresh = 0.9, dis_thres = chi2inv95[5]):
        if hsm.size == 0 or vsm.size == 0:return [], np.arange(hsm.shape[0]), np.arange(vsm.shape[1])
        hsm = hsm[:, :-1].copy()
        vsm = vsm[:-1, :].copy()
        confirm = (hsm > thresh) * (vsm > thresh) * (em > 0.8) * (distance > 0.5)
        my, mx = np.where(confirm)
        matches = np.stack([my, mx], axis = -1)
        return matches, np.delete(np.arange(hsm.shape[0]), my), np.delete(np.arange(vsm.shape[1]), mx)
        
    def match(self, cm, thresh):
        if cm.size == 0:
            return [], np.arange(cm.shape[0]), np.arange(cm.shape[1])
        matches = []
        _, x, y = lap.lapjv(cm, extend_cost = True, cost_limit = thresh)
        for ix, mx in enumerate(x):
            if mx >= 0:matches.append([ix, mx])
        u_tracks = np.where(x < 0)[0]
        u_dets = np.where(y < 0)[0]
        return matches, u_tracks, u_dets
    
    def iou_solver1(self, detection, thresh = 0.5):
        boxes = detection[:, :4]
        scores = detection[:, 4]
        return self.iou_solver(boxes, scores, thresh)
    
    def iou_solver(self, boxes, scores, thresh = 0.5):
        iou = np.zeros((boxes.shape[0], boxes.shape[0]), dtype = 'float32')
        for i, box in enumerate(boxes):
            if scores[i] > 0.6:continue
            iou[i, :] = KalmanFilter.iou(box, boxes)
            iou[i, i] = 0
        biou = iou > thresh
        sbiou = np.where(biou.sum(axis = 1) > 0)[0]
        if len(sbiou) == 0:
            return []
        for dn in range(1, sbiou.size + 1):
            cost = []
            res = []
            comb = [i for i in itertools.combinations(sbiou, dn)]
            for i in comb:
                tiou = iou.copy()
                c = 0
                rl = list(i)
                rl.sort(reverse = True)
                for j in rl:
                    tiou = np.delete(tiou, j, 0)
                    tiou = np.delete(tiou, j, 1)
                    c += scores[j]
                r = (tiou > thresh).sum()
                cost.append(c)
                res.append(r)
            if min(res) == 0:
                index = np.where(np.array(res)==0)[0]
                idx = np.argmin(np.array(cost)[index])
                dl = comb[index[idx]]
                return dl
        assert False
    
    def compose(self, dets, scores, feats):
        boxes = self.xywh2ccwh(dets)
        return np.concatenate([boxes, scores, feats], axis = -1)
    
    def getres(self, tracks, detection, matches):
        ress = []
        for trackid, detid in matches:
            track = tracks[trackid]
            det = detection[detid]
            if track.state == TrackState.Tracked:
                ress.append([*(det[:2] - track.prediction[:2]).tolist(), track.prediction[3]])
            
        if len(ress) > 0:
            ress = np.array(ress)
            res = 1. / ress[:, 2]
            weights = res / res.sum()
            res = ress[:, :2] * weights.reshape([-1, 1])
            return res.sum(axis = 0)
        return np.array(0)
     
    def update(self, tracks, detection, matches, frame_id, res = 0, strict = False):
         for trackid, detid in matches:
             track = tracks[trackid]
             det = detection[detid]
             if strict and track.state == TrackState.Lost and det[4] < 0.4:
                 track.forward(frame_id, res)
                 continue
             node = self.createNode(frame_id, det)
             track.update(node, res = res)
    
    def solver(self, img, frame_id = None):
        if frame_id >= 35:
            print()
        b, s, f, h, free = self.forward_detection(img)
        #visualize heatmap
        h_s = self.getsigmoid(h)
        hr_s = self.getsigmoid(free)
        h_s = cv2.applyColorMap((h_s[0]*255).astype(np.uint8), cv2.COLORMAP_JET)
        hr_s = cv2.applyColorMap((hr_s[0]*255).astype(np.uint8), cv2.COLORMAP_JET)
        cv2.imwrite('debug/hm_%04d.bmp'%frame_id, np.concatenate([h_s, hr_s], axis = 0))
        
        
        dets, scores, feats = self.DA.unmold(b[0], self.meta), s[0], f
        detection = self.compose(dets, scores, feats)
        confirmed_detection = detection[detection[:,4] >= self.config.DETECTION_MIN_CONFIDENCE, :]
        track_list = self.Tracked + self.Lost
        probes = np.array([track.getfeature() for track in track_list])
        gallery = confirmed_detection[:, 5:]
        
        em, oem, h_sim, v_sim = self.forward(probes, gallery)
        boxes1 = np.array([track.prediction for track in track_list])
        boxes2 = confirmed_detection[:, :4]
        distance = self.getdis(boxes1, boxes2)
        
        embedding_matches, u_tracks, u_dets = self.embedding_match(em, h_sim, v_sim, distance)
        res = self.getres(track_list, confirmed_detection, embedding_matches)
        #update distance
        if len(boxes1) > 0:
            boxes1[:, :2]+=res
            distance = self.getdis(boxes1, boxes2)
            
        self.update(track_list, confirmed_detection, embedding_matches, frame_id, res)
        matches = np.array(embedding_matches).tolist()
        
        detection = confirmed_detection[u_dets]
        tracks = np.array(track_list)[u_tracks]
        inita = np.ones((len(detection)))
        
        am = 1-oem
        am[am > 0.3]=np.inf
        am[distance < 0.9] = np.inf
        cm = am[u_tracks][:, u_dets]
        jde_matches, u_tracks_jde, u_dets_jde = self.match(cm, 0.7)
        self.update(tracks, detection, jde_matches, frame_id, res, strict = True)
        
        cm = (1-em[:, u_dets])
        dm = (1-distance[:, u_dets])
        inita[(cm < 0.3).sum(axis = 0) > 0] = 0
        inita[(dm < 0.05).sum(axis = 0) > 0] = 0
        
        for match in jde_matches:matches.append([u_tracks[match[0]], u_dets[match[1]]])
        self.backward(matches)
        self.backward_detection(matches, scores)
        
        detection = detection[u_dets_jde]
        tracks = tracks[u_tracks_jde]
        am = am[u_tracks_jde][:, u_dets_jde]
        inita = inita[u_dets_jde]
        
        for track in tracks:track.forward(frame_id, res)
        
        if len(self.New) > 0:
            new_probes = np.array([track.features[2][2] for track in self.New])
            new_gallery = detection[:, 5:]
            boxes1 = np.array([track.prediction for track in self.New])
            boxes1[:, :2]+=res
            boxes2 = detection[:, :4]
            ds = self.getdis(boxes1, boxes2)
            di = self.getsim(new_probes, new_gallery)
            di[ds < 0.9] = 0
        else:
            di = np.zeros((0, len(detection)), dtype = 'float32')
        niou_matches, u_tracks_new, u_dets_new = self.match(1 - di, 0.3)
        self.update(self.New, detection, niou_matches, frame_id, res)
        
        TT, LT, IT, RT = [], [], [], []
        
        for t in self.New + self.Tracked + self.Lost:
            if t.state == TrackState.Tracked:TT.append(t)
            elif t.state == TrackState.Lost:LT.append(t)
            elif t.state == TrackState.Invalid:IT.append(t)
            elif t.state == TrackState.Removed:RT.append(t)
            elif t.state == TrackState.New:RT.append(t)
                
        self.Tracked = TT
        self.Lost = LT
        self.Invalid.extend(IT)
        self.Removed.extend(RT)
        self.New = []
        
        inita = inita[u_dets_new]
        dl = self.iou_solver1(detection[u_dets_new])
        u_dets_new = [d for i, d in enumerate(u_dets_new) if (i not in dl) and inita[i] > 0]
        
        for detid in u_dets_new:
            if detection[detid][4] < 0.5:continue
            node = self.createNode(frame_id, detection[detid])
            self.New.append(Trajectory(node, -1, self.config, self.seqInfo))
            
        self.appendResult(frame_id)
            
    
