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


if __name__ == '__main__':
    rospy.init_node('om_test', anonymous = False)
    print("connecting")
    scene_count = 0
    scene_limit = 500
    client = MongoClient()
    client = MongoClient('localhost', 62345)
    targets = []
    db = client.soma2data['soma2_pre_21_july']
    num = 0
    sz = 0
    for entry in db.find():
        id = entry['id']
        if("-" in id):
            clsz = entry['cloud']['width']
            print(clsz)
            if(clsz < 250 or clsz > 15000):
                continue
            num+=1
            sz+=clsz
    avg = sz/num
    print("avg: "+str(avg))
