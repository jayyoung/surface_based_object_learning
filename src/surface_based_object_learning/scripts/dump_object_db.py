#!/usr/bin/env python
import roslib
import rospy
from sensor_msgs.msg import PointCloud2, PointField
from mongodb_store.message_store import MessageStoreProxy
from sensor_msgs.msg import PointCloud2, Image
from cv_bridge import CvBridge, CvBridgeError
from soma_llsd_msgs.msg import *
from soma_msgs.msg import SOMAObject
from soma_manager.srv import *
from soma_llsd.srv import *
from sensor_msgs.msg import Image, PointCloud2, CameraInfo, JointState
import cv
import cv2
import os
import pickle
import python_pcd
import sys
#from object_interestingness_estimator.srv import *
import caffe
import scipy.misc
#from PIL import Image
import numpy as np
#import pandas as pd
import cPickle
#import logging
import base64
import datetime, time
import sys
import numpy as np
#import cv2




if __name__ == '__main__':
    rospy.init_node('om_test', anonymous = False)

    bridge = CvBridge()
    print("beginning")
    soma_query_service = rospy.ServiceProxy('/soma/query_objects',SOMAQueryObjs)
    segment_query_service = rospy.ServiceProxy('/soma_llsd/get_segment',GetSegment)
    scene_query_service = rospy.ServiceProxy('/soma_llsd/get_scene',GetScene)
    query = SOMAQueryObjsRequest()
    query.query_type = 0
    query.objecttypes=['unknown']
    response = soma_query_service(query)
    #rospy.wait_for_service('/object_interestingness_estimator/estimate',10)
    #interest_srv = rospy.ServiceProxy('/object_interestingness_estimator/estimate',EstimateInterest)
    bridge = CvBridge()
    #CNNWrapper.default_args.update({'gpu_mode': True})
    #c = CNNWrapper(**CNNWrapper.default_args)
    sift = cv2.SURF(1000)
    print("processing objects")
    for k in response.objects:
        print("getting: " + k.id)
        object_target_dir = "object_dump/"+str(eval(k.metadata)['waypoint'])+"/"+k.id+"/"
        #print("writing " + object_target_dir)
        segment_req = segment_query_service(k.id)
        print("getting seg imgs")

        for obs in segment_req.response.observations:
            #scene = scene_query_service(obs.scene_id)
            #scene_rgb = bridge.imgmsg_to_cv2(scene.response.rgb_img)
            rgb = obs.rgb_cropped
            cv_rgb_image = bridge.imgmsg_to_cv2(rgb)
            #cv_mask_image = bridge.imgmsg_to_cv2(obs.image_mask)
            #height, width, depth = cv_mask_image.shape
            #cv_mask_image = cv2.cvtColor(cv_mask_image,cv2.COLOR_RGB2GRAY)
            #_,thresh = cv2.threshold(cv_mask_image,1,255,cv2.THRESH_BINARY)
            #cv_mask_image = cv2.convertScaleAbs(cv_mask_image)
            #print(cv_mask_image.shape)
            ##print(cv_rgb_image.shape)
            #res = cv2.bitwise_and(scene_rgb,scene_rgb,mask = cv_mask_image)
            #contours,hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            #cnt = contours[0]
            #x,y,w,h = cv2.boundingRect(cnt)
            #crop = res[y:y+h,x:x+w]
            kp, des = sift.detectAndCompute(cv_rgb_image,None)
            print("kp:" + str(len(kp)))
            if(len(kp) < 20):
                continue
            #interest_points = interest_srv(obs.map_cloud)
            #print("des:" + str(len(des)))
            #if(interest_points.output.data >= 4):
            if not os.path.exists(object_target_dir):
                os.makedirs(object_target_dir)
            print("--- WRITING ---")
            print("accepting")
            cv2.imwrite(object_target_dir+str(segment_req.response.observations.index(obs))+"-"+str(len(kp))+".png",cv_rgb_image)



    print("all done!")
