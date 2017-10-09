#!/usr/bin/env python
import roslib
import rospy
from sensor_msgs.msg import PointCloud2, PointField
from mongodb_store.message_store import MessageStoreProxy
from sensor_msgs.msg import PointCloud2, Image
from cv_bridge import CvBridge, CvBridgeError
from soma_llsd_msgs.msg import *
import cv
import cv2
import os
import pickle
from pymongo import MongoClient
import python_pcd

if __name__ == '__main__':
    rospy.init_node('om_test', anonymous = False)
    print("connecting")
    target = "scene_dump/1c40eabb-3084-47b1-9bff-84272564c4f0"

    for d,s,f in os.walk(target):
        for k in f:
            print("loading " + str(k))
            of = open(d+"/"+k)
            view = pickle.load(of)
            print("done")
            python_pcd.write_pcd(view.id+".pcd",view.cloud)
