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

    learning_episodes = 0
    views_taken = 0
    for d,s,f in os.walk("scene_dump/"):
        if("-" in d):
            learning_episodes+=1
            for k,j,l in os.walk(d):
                views_taken+=len(l)

    print("learning episodes: " + str(learning_episodes))
    print("views taken: " + str(views_taken))
