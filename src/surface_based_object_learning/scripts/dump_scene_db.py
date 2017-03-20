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
    scene_limit = 5
    client = MongoClient()
    client = MongoClient('localhost', 62345)
    targets = []
    db = client.somadata['llsd_scene_store']
    for post in db.find({"meta_data": "{\"source\": \"surface_based_object_learning\"}"}):
        targets.append(post['id'])
        if(scene_count > scene_limit):
            print("limit reached, breaking")
            break

    print("done finding ids")

    bridge = CvBridge()
    print("trying to connect to database")
    store = MessageStoreProxy(database="somadata", collection="llsd_scene_store")
    processed_episodes = []
    for idd in targets:
        print("running query")
        scenes = store.query_named(idd,Scene._type)
        targ = "scene_dump/"
        print("got some scenes." + str(len(scenes)) +" in fact! writing them to " + targ)
        if not os.path.exists(targ):
            os.makedirs(targ)
        for sc in scenes:
            cur_scene = sc
            if(hasattr(cur_scene,'meta_data')):
                md = eval(cur_scene.meta_data)
                if "surface" in md['source']:
                    print("processing: " + cur_scene.id)
                    if not os.path.exists(targ+cur_scene.episode_id+"/"):
                        os.makedirs(targ+cur_scene.episode_id+"/")
                    if(cur_scene.episode_id not in processed_episodes):
                        processed_episodes.append(cur_scene.episode_id)
                    scene_count+=1
                    pickle.dump(cur_scene,open(targ+cur_scene.episode_id+"/"+cur_scene.id+".p",'wb'))
        if(scene_count > scene_limit):
            print("limit reached, breaking")
            break

    print("processed: " + str(scene_count) + " scenes and " + str(len(processed_episodes)) + " episodes")

    print("all done!")
