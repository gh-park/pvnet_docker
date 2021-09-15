
from numpy.ctypeslib import ndpointer
import argparse
import json
import os
import time
import numpy as np
import math
import threading
import time
import rospy
from geometry_msgs.msg import Pose
from visualization import Visualizer2D as vis

from lib.config import cfg, args
from lib.networks import make_network
from lib.datasets import make_data_loader
from lib.utils.net_utils import load_network
from numpy.ctypeslib import ndpointer

#-----------------------------------------------------------------------------

#import matplotlib.pyplot as plt
import ctypes
import numpy as np
import cv2
import os
import threading
import tqdm
import torch
from pyquaternion import Quaternion
from lib.visualizers import make_visualizer
import pycocotools.coco as coco
from lib.datasets import make_data_loader
from lib.evaluators import make_evaluator
import tqdm
import torch
from lib.utils.pvnet import pvnet_pose_utils
from lib.networks import make_network
import matplotlib.patches as patches
from lib.utils.net_utils import load_network
from ctypes import cdll
import pyrealsense2 as rs
#---------------------------------------realsense---------------------------------------------
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pose_pred=np.zeros((4,4))
def detect_img():

    global pose_pred
    pose_pred=np.eye(4)
    def nothing(x):
       pass

    mask = np.zeros((640,480),dtype=np.uint8)
    cv2.namedWindow("rgb", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("rgb", 640,480)
    cv2.namedWindow("mask", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("mask", 640,480)
    line_arr1 = [0, 1, 3, 2, 0, 4, 6, 2]
    line_arr2 = [5, 4, 6, 7, 5, 1, 3, 7]
    pipeline.start(config)

    network = make_network(cfg).cuda()
    load_network(network, cfg.model_dir, epoch=cfg.test.epoch)
    network.eval()
    visualizer = make_visualizer(cfg)
    data_loader = make_data_loader(cfg, is_train=False)
    evaluator = make_evaluator(cfg)
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]

    fps_3d=np.load("fps_3d.npy")
    center_3d=np.load("center_3d.npy")
    corner_3d=np.load("corner_3d.npy")
    prev_pose_pred=np.zeros((4,4))
   # K = np.array([[700.0, 0.0, 320.0], [0.0, 700.0, 240.0], [0.0, 0.0, 1.0]])
    #K = np.array([[2.40691932e+03, 0.00000000e+00, 3.19065470e+02],[0.00000000e+00, 2.71622117e+03, 2.03018751e+02],[0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
    K = np.array([[608.13312805 ,  0.      ,   324.97036511],[  0.     ,    612.61549006 ,242.14119643],[  0.       ,    0.        ,   1.        ]])

    while 1:
     start = time.time()
     try:
         print("Detect IMG")
         frames = pipeline.wait_for_frames(50)
         color_frame = frames.get_color_frame()
         if not color_frame:
            print("Frame ERROR")
            continue
         color_img = np.array(color_frame.get_data())
         print(color_img.shape)
         color_img_= (cv2.resize(color_img,(640,480),interpolation=cv2.INTER_CUBIC)/255-mean)/std
         color_img__ = np.zeros((1,3,480,640),dtype=float)
         color_img__[0,2,:,:] = color_img_[:,:,0]
         color_img__[0,1,:,:] = color_img_[:,:,1]
         color_img__[0,0,:,:] = color_img_[:,:,2]
         inp = torch.reshape(torch.from_numpy(color_img__),(1,3,480,640)).type(torch.FloatTensor).cuda()
         with torch.no_grad():
          try:
             output = network(inp)
             print("output")
          except Exception as inst:
             print("NETWORK ERROR")
             print(inst)
             pass
         #print(output['mask'].shape)
         kpt_2d = output['kpt_2d'][0].detach().cpu().numpy()
        
         kpt_3d = np.concatenate([fps_3d, [center_3d]], axis=0)

         #K = np.array([[700.0, 0.0, 320.0], [0.0, 700.0, 240.0], [0.0, 0.0, 1.0]])
         alpha = 0.99
         now_pose_pred = pvnet_pose_utils.pnp(kpt_3d, kpt_2d, K)
         pose_pred = now_pose_pred 
        # for j in range(0,3):
        #    for k in range(0,4):
        #       pose_pred[j,k] = 0.0*prev_pose_pred[j,k]+1*now_pose_pred[j,k]
        
        # prev_pose_pred = now_pose_pred
         #print("pose_pred ::::: \n",pose_pred)
         center3d = [[np.mean(corner_3d[:,0]),np.mean(corner_3d[:,1]),np.mean(corner_3d[:,2])],
[np.mean(corner_3d[:,0])+0.05,np.mean(corner_3d[:,1]),np.mean(corner_3d[:,2])],
[np.mean(corner_3d[:,0]),np.mean(corner_3d[:,1])-0.05,np.mean(corner_3d[:,2])],
[np.mean(corner_3d[:,0]),np.mean(corner_3d[:,1]),np.mean(corner_3d[:,2])+0.05]]
         corner_2d_pred = pvnet_pose_utils.project(corner_3d, K, pose_pred)
         center_2d_pred = pvnet_pose_utils.project(center3d, K, pose_pred)

         showimg= np.uint8((color_img_*std+mean)*255)
         showimg_= np.zeros((480,640,3),dtype=np.uint8)
         showimg_[:,:,0] = showimg[:,:,2]
         showimg_[:,:,1] = showimg[:,:,1]
         showimg_[:,:,2] = showimg[:,:,0]

         mask = np.transpose(np.transpose(np.uint8(np.reshape(output['mask'][0].detach().cpu().numpy(),(480,640))))*255)

         try:
            show_img = cv2.resize(color_img.copy(),(640,480),interpolation=cv2.INTER_CUBIC)
            for i in range(len(line_arr1)-1):
               show_img = cv2.line(show_img, (int(corner_2d_pred[line_arr1[i]][0]),int(corner_2d_pred[line_arr1[i]][1])),(int(corner_2d_pred[line_arr1[i+1]][0]),int(corner_2d_pred[line_arr1[i+1]][1])),(255,0,0),1)
            for i in range(len(line_arr2)-1):
               show_img = cv2.line(show_img, (int(corner_2d_pred[line_arr2[i]][0]),int(corner_2d_pred[line_arr2[i]][1])),(int(corner_2d_pred[line_arr2[i+1]][0]),int(corner_2d_pred[line_arr2[i+1]][1])),(255,0,0),1)
            #print(center_2d_pred[0])
            show_img = cv2.circle(show_img, (int(center_2d_pred[0][0]),int(center_2d_pred[0][1])),1,(0,255,0),3)
            show_img = cv2.line(show_img, (int(center_2d_pred[0][0]),int(center_2d_pred[0][1])),(int(center_2d_pred[1][0]),int(center_2d_pred[1][1])),(255,0,0),2)
            show_img = cv2.line(show_img, (int(center_2d_pred[0][0]),int(center_2d_pred[0][1])),(int(center_2d_pred[2][0]),int(center_2d_pred[2][1])),(0,255,0),2)
            show_img = cv2.line(show_img, (int(center_2d_pred[0][0]),int(center_2d_pred[0][1])),(int(center_2d_pred[3][0]),int(center_2d_pred[3][1])),(0,0,255),2)
            try:
               ret, img_binary = cv2.threshold(mask, 127, 255, 0)
               contours, color_hierachy =cv2.findContours(img_binary.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
               c0 = np.reshape(contours[0],(-1,2))
               show_img = cv2.drawContours(np.uint8(show_img), contours, -1, (0,255,0), 1)
            except:
               print("ERRRRRRRR")
               pass
            show_img_resize = cv2.resize( show_img.copy(),(640,480),interpolation=cv2.INTER_CUBIC)
            cv2.imshow("rgb",show_img_resize)
         except:
             cv2.imshow("rgb",color_img)


         seg_mask =cv2.resize(mask.copy(),(640,480),interpolation=cv2.INTER_CUBIC)
         cv2.imshow("mask",seg_mask)

         k = cv2.waitKey(5) & 0xFF
         if k == ord('s'):
             cv2.destroyWindow("rgb")
     except:
         print("IMG ERROR")
     if(time.time()-start<0.05):
        time.sleep(0.05-(time.time()-start))
     print("IMG TIME :",time.time()-start) 
    end()
    evaluator.summarize()

'''
     try:
      seg_mask =cv2.resize(mask.copy(),(640,480),interpolation=cv2.INTER_CUBIC)

      cv2.imshow("rgb",show_img_resize)

      cv2.imshow("mask",seg_mask)
      k = cv2.waitKey(5) & 0xFF
      if k == ord('s'):
         cv2.destroyWindow("rgb")
     except Exception as inst:
      print(inst)
      cv2.imshow("rgb",color_img)
      cv2.imshow("mask",seg_mask)
      k = cv2.waitKey(5) & 0xFF
      if k == ord('s'):
         cv2.destroyWindow("rgb")
      pass
'''   
if __name__ == '__main__':
    pose_pred=np.eye(4)
    t1 = threading.Thread(target=detect_img)
    t1.start()
    pub = rospy.Publisher('obj_pose', Pose, queue_size=1)
    rospy.init_node('obj_pose_publisher', anonymous=True)
    rate = rospy.Rate(20) # Hz
    while(1):
        try:
            q8d = Quaternion(matrix=pose_pred[0:3,0:3])
            p = Pose()
            p.position.x = pose_pred[0,3];
            p.position.y = pose_pred[1,3];
            p.position.z = pose_pred[2,3];
            p.orientation.x=q8d[0];
            p.orientation.y=q8d[1];
            p.orientation.z=q8d[2];
            p.orientation.w=q8d[3];
            pub.publish(p)
        except:
            print("QUAT ERROR")
            pass            

        rate.sleep()



