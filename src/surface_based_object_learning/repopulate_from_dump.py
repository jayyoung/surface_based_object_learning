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
from object_learning_core import *
import tf2_ros
import tf, tf2_msgs.msg
import python_pcd

if __name__ == '__main__':
    #rospy.init_node('DUMP_POPULATOR', anonymous = False)

    we = LearningCore("localhost","62345","~/surface_based_object_learning_rgb_dump/","Common")
    cip = rospy.Publisher("/head_xtion/depth_registered/sw_registered/camera_info",CameraInfo)
    tip = rospy.Publisher("/tf",tf2_msgs.msg.TFMessage)
    got_counter = False
    with open("completed.txt", "w") as completed:

        for subdir, dirs, files in os.walk("scripts/scene_dump"):
            print("--episode--")
            we.begin_obs(None)
            for k in files:
                print("PROCESSING VIEW "+k)
                try:
                    loaded = pickle.load( open(subdir+"/"+k, "rb" ))
                    #if("173aea17-a250-497e-8824-351e2bf5817e" == loaded.id):
                    #    print("FOUND IT")
                    #got_counter = True
                    observation_structure = {}
                    observation_structure['rgb_image'] = loaded.rgb_img
                    observation_structure['camera_info'] = loaded.camera_info
                    observation_structure['scene_cloud'] = loaded.cloud
                    #python_pcd.write_pcd("cloud.pcd",loaded.cloud)
                    observation_structure['robot_pose'] = loaded.robot_pose
                    observation_structure['metadata'] = loaded.meta_data
                    print("processing wp: " + loaded.meta_data +" scene id: " + loaded.id)
                    #if("Counter" not in loaded.waypoint):
                    #    continue
                    observation_structure['tf'] = loaded.transform
                    observation_structure['depth_image'] = loaded.depth_img
                    observation_structure['timestamp'] = loaded.timestamp
                    rospy.Rate(10)
                    for i in range(20):
                        cip.publish(loaded.camera_info)
                        rospy.sleep(0.1)
                        tip.publish(loaded.transform)
                        rospy.sleep(0.1)
                    we.process_scene(loaded.cloud,loaded.waypoint,observation_structure)
                    completed.write(k)
                except Exception,e:
                    print("failed to process file: " + str(k) + " for some reason")
                    print(e)
            we.end_obs(None)
            #if(got_counter):
            #    print("got the kitchen, finishing")
            #    break
