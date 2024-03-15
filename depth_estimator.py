# !/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
import pcl.pcl_visualization
import pcl
import os
import sys

# ModuleNotFoundError: No module named '' 解决办法
parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 2)))
scripts_path = os.path.join(parent_path, "scripts")
sys.path.append(scripts_path)
clrnet_model_path = os.path.join(parent_path, "det_clrnet")
sys.path.append(clrnet_model_path)


""" ============ 单目深度估计算法 """
class DepthEstimator(object):
    def __init__(self, K, Hc):        
        # K = [  [fu   0  cu]   
        #        [ 0  fv  cv]  
        #        [ 0   0   1]  ]
        self.K = K
        self.K_inv = np.linalg.inv(K)
        self.Hc = Hc
        
        # 3*4
        self.K_concat = np.concatenate((K, np.array([[0, 0, 0]]).transpose()), axis=1)
        # K_concat = [  [fu   0  cu  0]   
        #               [ 0  fv  cv  0]  
        #               [ 0   0   1  0]  ]
        self.params_init(self.K_concat)
        return 
        

    def params_init(self, K_concat):
        # pitch 
        alpha = np.deg2rad(-3.40)

        # roll
        gamma = np.deg2rad(0.82)
        
        '''初始内外参'''
        self.cu, self.cv = K_concat[0,2], K_concat[1,2]
        self.fu, self.fv = K_concat[0,0], K_concat[1,1]
        self.bx = -K_concat[0,3] / self.fu
        
        self.sin_alpha = np.sin(alpha) 
        self.cos_alpha = np.cos(alpha) 
        self.sin_gamma = np.sin(gamma) 
        self.cos_gamma = np.cos(gamma)
        return


    def callback(self, uu, vv):
        # 深度估计
        depth_pred = (self.Hc - self.sin_gamma * self.bx) / (- self.cos_gamma * self.sin_alpha + 
                            self.sin_gamma * (uu - self.cu) / self.fu + 
                               self.cos_gamma * self.cos_alpha * (vv - self.cv) / self.fv)
          
        ### convert bounding box center (u,v) to camera coordinates
        tmp = np.dot(  np.array(self.K_inv), np.array([[uu, vv, 1]]).transpose()  )
        # print("  tmp", tmp, tmp[0][0], tmp[1][0], tmp[2][0])

        xc = tmp[0][0] * depth_pred
        yc = tmp[1][0] * depth_pred
        zc = tmp[2][0] * depth_pred   
        # print("xc, yc, zc ", xc[0], yc[0], zc[0])     
        return xc, yc, zc
