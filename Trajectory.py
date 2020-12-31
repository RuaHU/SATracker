#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 09:51:50 2019

@author: hu
"""
import numpy as np

class TrackState(object):
    New = 0
    Tracked = 1
    Lost = 2
    Invalid = 3
    Removed = 4

class Trajectory():
    count = 0
    def __init__(self, node, ID = None, config = None, seqInfo = None):
        self.config = config
        self.seqInfo = seqInfo
        #self.ih, self.iw = seqInfo['imHeight'], seqInfo['imWidth']
        self.is_activated = False
        self.activate(node)
        self.max_time_lost = 30
        self.max_score = 0.

    @staticmethod
    def alloc_id():
        Trajectory.count += 1
        np.random.seed(Trajectory.count)
        return Trajectory.count, np.random.randint(0, 255, 3)
        
    def activate(self, node):
        self.id = -1
        self.kf = KalmanFilter()
        self.kf.update(node.association_dict['box'])
        self.T = [node]
        self.frame_id = node.frame_id
        self.features = [[0, 0, node.association_dict['feature']],
                         [0, 0, node.association_dict['feature']],
                         [0, 0, node.association_dict['feature']]]
        self.state = TrackState.New
        
    def getfeature(self,):
        if self.state == TrackState.Tracked:
            return self.features[1][2]
        else:
            return self.features[2][2]
        
    def re_activate(self, node, new_id = False):
        self.kf.update(node.association_dict['box'])
        self.T.append(node)
        self.update_feature(node)
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = node.frame_id
        if new_id: self.id, self.c = self.alloc_id()
            
    def update(self, node, res = 0, update_feature = True):
        if self.id == -1:self.id, self.c = self.alloc_id()
        self.frame_id = node.frame_id
        self.T.append(node)
        self.kf.update(node.association_dict['box'], res)
        if node.association_dict['score'] > self.max_score:
            self.max_score=node.association_dict['score']
        self.state = TrackState.Tracked
        self.is_activated = True
        if update_feature:self.update_feature(node)
        
    def forward(self, frame_id, res = 0):
        self.prediction[:2] += res
        self.kf.update(self.prediction)
        self.mark_lost()
        if frame_id - self.frame_id > self.max_time_lost:self.mark_invalid()
        
    def mark_lost(self,):
        self.state = TrackState.Lost
    
    def mark_invalid(self,):
        self.state = TrackState.Invalid
    
    def mark_removed(self,):
        self.state = TrackState.Removed
    
    def update_feature(self, node):
        if node.association_dict['feature'] is None : return
        self.features[0] = [0, 0, node.association_dict['feature']]
        if not node.association_dict['occlusion']:self.features[1] = self.features[0]
        feat = self.features[2][2]
        smooth_feat = 0.9 * feat + 0.1 * node.association_dict['feature']
        smooth_feat /= np.linalg.norm(smooth_feat)
        self.features[2] = [0, 0, smooth_feat]
        smooth_feat = 0.5 * feat + 0.5 * node.association_dict['feature']
        smooth_feat /= np.linalg.norm(smooth_feat)
        self.features[1] = [0, 0, smooth_feat]
        
    def predict(self, interval = 1):
        self.prediction = self.kf.predict(interval)
        return self.prediction
    
class Node():
    def __init__(self, frame_id, association_dict = None):
        '''
        box_id: list [id1, id2]
        box: list [box1, box2]
        '''
        self.frame_id = frame_id
        self.association_dict = association_dict
        
    def __eq__(self, node):
        if self.frame_id == node.frame_id and (self.box_id[0] in node.box_id) and (self.box[0] in node.box):
            return True
        else:
            return False
    
    def optimalboxtocycx(self, factor = 1.):
        return [(self.optimal_box[1] + self.optimal_box[3] * 0.5) * factor, (self.optimal_box[0] + self.optimal_box[2] * 0.5) * factor]
    
    def optimalboxtoyxyx(self,):
        return [self.optimal_box[1], self.optimal_box[0], self.optimal_box[1] + self.optimal_box[3], self.optimal_box[0] + self.optimal_box[2]]
    
# vim: expandtab:ts=4:sw=4
import scipy.linalg

"""
Table for the 0.95 quantile of the chi-square distribution with N degrees of
freedom (contains values for N=1, ..., 9). Taken from MATLAB/Octave's chi2inv
function and used as Mahalanobis gating threshold.
"""
chi2inv95 = {
    1: 3.8415,
    2: 5.9915,
    3: 7.8147,
    4: 9.4877,
    5: 11.070,
    6: 12.592,
    7: 14.067,
    8: 15.507,
    9: 16.919}
        
class KalmanFilter(object):
    def __init__(self):
        ndim, dt = 4, 1.

        # Create Kalman filter model matrices.
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)

        # Motion and observation uncertainty are chosen relative to the current
        # state estimate. These weights control the amount of uncertainty in
        # the model. This is a bit hacky.
        self._std_weight_position = 1. / 20
        self._std_weight_velocity = 1. / 160
        
        self.initialed = False

    def initiate(self, measurement):
        """Create track from unassociated measurement.

        Parameters
        ----------
        measurement : ndarray
            Bounding box coordinates (x, y, a, h) with center position (x, y),
            aspect ratio a, and height h.

        Returns
        -------

        """
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        self.mean = np.r_[mean_pos, mean_vel]
        
        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3]]
        self.covariance = np.diag(np.square(std))

    def predict(self, interval = 1):
        prediction = []
        mean = self.mean.copy()
        for i in range(interval):
            mean = np.dot(mean, self._motion_mat.T)
            prediction.append(mean[:4])
        for mean in prediction:
            mean[2:3] *= mean[3:]
        if interval == 1:return prediction[0]
        return prediction

    def project(self, update = False):
        std_pos = [
            self._std_weight_position * self.mean[3],
            self._std_weight_position * self.mean[3],
            1e-2 * np.ones_like(self.mean[3]),
            self._std_weight_position * self.mean[3]]
        std_vel = [
            self._std_weight_velocity * self.mean[3],
            self._std_weight_velocity * self.mean[3],
            1e-5 * np.ones_like(self.mean[3]),
            self._std_weight_velocity * self.mean[3]]
        sqr = np.square(np.r_[std_pos, std_vel]).T

        motion_cov = np.diag(sqr)

        mean = np.dot(self.mean, self._motion_mat.T)
        
        left = np.dot(self._motion_mat, self.covariance)
        covariance = np.dot(left, self._motion_mat.T) + motion_cov
        if update:self.mean, self.covariance = mean, covariance
        
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3]]
        innovation_cov = np.diag(np.square(std))

        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot((
            self._update_mat, covariance, self._update_mat.T))
        return mean, covariance + innovation_cov

    @staticmethod
    def toxyxy(m):
        m[..., :2] -= 0.5 * m[..., 2:]
        m[..., 2:] += m[..., :2]
        return m

    
    def update(self, box, res = 0, update_iou = True, factor = 1):
        measurement = box.copy()
        measurement[2] /= measurement[3]
        if self.initialed == False:
            self.initiate(measurement)
            self.initialed = True
            return
        
        self.mean[:2]+=res
        projected_mean, projected_cov = self.project(True)

        chol_factor, lower = scipy.linalg.cho_factor(
            projected_cov, lower=True, check_finite=False)
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower), np.dot(self.covariance, self._update_mat.T).T,
            check_finite=False).T
        innovation = (measurement - projected_mean) * factor

        self.mean = self.mean + np.dot(innovation, kalman_gain.T)
        self.covariance = self.covariance - np.linalg.multi_dot((
            kalman_gain, projected_cov, kalman_gain.T))

