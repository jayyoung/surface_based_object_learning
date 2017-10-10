#!/usr/bin/env python

# general stuff
import roslib
import rospy
import sys
import argparse
import os
from random import randint
import cv2
import json
# view store STUFF
from mongodb_store.message_store import MessageStoreProxy
from soma_llsd_msgs.msg import Segment,Observation,Scene
from cv_bridge import CvBridge, CvBridgeError
# ROS stuff
from sensor_msgs.msg import PointCloud2, PointField
from segment_processing import SegmentProcessor
#from recognition_manager import ObjectRecognitionManager
from view_registration import ViewAlignmentManager
from sensor_msgs.msg import Image, PointCloud2, CameraInfo, JointState
from std_srvs.srv import Trigger, TriggerResponse
from geometry_msgs.msg import Pose,Point,Quaternion
import tf2_ros
import tf, tf2_msgs.msg
import os
# WS stuff
from surface_based_object_learning.srv import *
from util import TransformationStore

# soma stuff
from soma_msgs.msg import *
from soma_manager.msg import *
from soma_manager.srv import *
from soma_llsd.srv import *

# recog stuff
#from recognition_srv_definitions.srv import *

import uuid
#import python_pcd

class LearningCore:

    #core = LearningCore(roitag,datadir,pc,depth,rgb,tf,pose,cam)

    def __init__(self,roi_tag,data_dump_dir,pc,depth,rgb,tf,pose,cam):
        rospy.init_node('surface_based_object_learning', anonymous = False)
        self.setup_clean = False
        rospy.loginfo("LEARNING CORE: Manager Online")
        # make a segment tracker
        rospy.loginfo("LEARNING CORE: setting up topics as provided")


        self.pointcloud_topic = pc
        self.rgb_image_topic = rgb
        self.depth_image_topic = depth
        self.tf_topic = tf
        self.pose_topic = pose
        self.camera_info_topic = cam
        self.roi_tag = roi_tag
        rospy.loginfo("pc "+pc)
        rospy.loginfo(rgb)
        rospy.loginfo(depth)
        rospy.loginfo("tf "+tf)
        rospy.loginfo(cam)
        rospy.loginfo(pose)
        rospy.loginfo("tags:")
        rospy.loginfo(roi_tag)
        rospy.loginfo(data_dump_dir)
        rospy.Rate(10) # 10hz



        self.data_dump_dir = data_dump_dir
        rospy.loginfo("setting up segment processor")
        self.segment_processor = SegmentProcessor()
        self.pending_obs = []
        self.cur_sequence_obj_ids = []
        self.cur_view_soma_ids = []
        self.cur_observation_data = None
        self.queued_soma_objs = []
        self.cur_scene_list = []
        self.just_data_collection = False
        self.cv_bridge = CvBridge()

        rospy.loginfo("LEARNING CORE: setting up services")
        process = rospy.Service('/surface_based_object_learning/process_scene',ProcessScene,self.process_scene_callback)
        rospy.loginfo("LEARNING CORE: scene processing service running")

        begin_observations = rospy.Service('/surface_based_object_learning/begin_observation_sequence',Trigger,self.begin_obs)
        end_observations = rospy.Service('/surface_based_object_learning/end_observation_sequence',Trigger,self.end_obs)

        rospy.loginfo("LEARNING CORE: setting up SOMA services")
        rospy.loginfo("LEARNING CORE: getting SOMA insert service")
        rospy.wait_for_service('soma/insert_objects')
        rospy.loginfo("LEARNING CORE: done")
        self.soma_insert = rospy.ServiceProxy('soma/insert_objects',SOMAInsertObjs)

        rospy.loginfo("LEARNING CORE: getting SOMA query service")
        rospy.wait_for_service('soma/query_objects')
        rospy.loginfo("LEARNING CORE: done")
        self.soma_get = rospy.ServiceProxy('soma/query_objects',SOMAQueryObjs)

        rospy.loginfo("LEARNING CORE: getting SOMA update service")
        rospy.wait_for_service('/soma/update_object')
        rospy.loginfo("LEARNING CORE: done")
        self.soma_update = rospy.ServiceProxy('soma/update_object',SOMAUpdateObject)

    #    rospy.loginfo("LEARNING CORE: setting up view alignment manager")
    #    self.view_alignment_manager = ViewAlignmentManager()

        rospy.loginfo("LEARNING CORE: getting LLSD services")
        rospy.wait_for_service('/soma_llsd/insert_scene')
        self.view_store_insert = rospy.ServiceProxy('/soma_llsd/insert_scene',InsertScene)

        rospy.wait_for_service('/soma_llsd/get_segment')
        self.get_segment = rospy.ServiceProxy('/soma_llsd/get_segment',GetSegment)

        rospy.wait_for_service('/soma_llsd/insert_segment')
        self.insert_segment = rospy.ServiceProxy('/soma_llsd/insert_segment',InsertSegment)

        rospy.wait_for_service('/soma_llsd/add_observations_to_segment')
        self.append_obs_to_segment = rospy.ServiceProxy('/soma_llsd/add_observations_to_segment',AddObservationsToSegment)

        self.scene_publisher = rospy.Publisher('/surface_based_object_learning/scenes', std_msgs.msg.String, queue_size=10)
        self.data_dump_publisher = rospy.Publisher('/surface_based_object_learning/data_dumps', std_msgs.msg.String, queue_size=10)
        self.obj_discovery_publisher = rospy.Publisher("/surface_based_object_learning/object_discovery",SOMANewObjects,queue_size=10)
        self.segment_bbox_publisher = rospy.Publisher("/surface_based_object_learning/segment_bbox_images",Image,queue_size=10)


        test = SOMANewObjects()
        test.ids.append("ping")
        self.obj_discovery_publisher.publish(test)
        rospy.sleep(1)
        self.obj_discovery_publisher.publish(test)
        rospy.sleep(1)
        self.obj_discovery_publisher.publish(test)

        self.clean_up_obs()

        rospy.loginfo("LEARNING CORE: -- node setup completed --")
        self.setup_clean = True

        rospy.spin()
    def begin_spinning(self):
        rospy.spin()

    def clean_up_obs(self):
        rospy.loginfo("LEARNING CORE: running cleanup")
        self.pending_obs = []
        self.cur_proc_scene = None
        self.cur_sequence_obj_ids = []
        self.cur_view_soma_ids = []
        self.cur_scene_list = []
        self.cur_observation_data = None
        self.just_data_collection = False
        self.segment_processor.reset()
        self.cur_episode_id = str(uuid.uuid4())
        self.queued_soma_objs = []
        rospy.loginfo("LEARNING CORE: -- new episode id: " + self.cur_episode_id)


    def begin_obs(self,req):
        rospy.loginfo("LEARNING CORE: -- received signal to begin sequence of observations --")
        if(self.setup_clean):
            rospy.loginfo("LEARNING CORE: ready to go")
        else:
            rospy.loginfo("LEARNING CORE: ERROR: node setup not completed yet, wait a sec and try again")
            return
        self.clean_up_obs()
        return TriggerResponse(True,"Observations Beginning: Assuming all subsequent observations are from the same sequence.")

    def end_obs(self,req):
        rospy.loginfo("LEARNING CORE: -- received signal to terminate sequence of observations --")
        rospy.loginfo("LEARNING CORE: ")
        self.do_postprocessing()
        self.clean_up_obs()
        return TriggerResponse(True,"Observations Ending: Assuming all previous observations were from the same sequence.")

    def flush_observation(self,data):
        print("-- flushing observation from dataset through system --")
        self.process_scene(data['cloud'],data['data'][3],data)

    def register_with_view_store(self,cloud,extra_data=None):
        try:
            # used in the case of offline data
            if(self.cur_observation_data is None):
                self.populate_observation_data(cloud,extra_data)

            # INSERT INTO THE VIEW STORE
            scene = self.view_store_insert(self.cur_episode_id,
            self.cur_observation_data['waypoint'],
            self.cur_observation_data['metadata'], # meta_data
            self.cur_observation_data['timestamp'],
            self.cur_observation_data['tf'],
            self.cur_observation_data['scene_cloud'],
            self.cur_observation_data['rgb_image'],
            self.cur_observation_data['depth_image'],
            self.cur_observation_data['camera_info'],
            self.cur_observation_data['robot_pose'])

            if(scene.result is True):
                rospy.loginfo("LEARNING CORE: successfully added scene to view store")
                self.cur_scene_id = scene.response.id
                self.cur_scene_list.append(self.cur_scene_id)

                rospy.loginfo("-- Writing Scene to HDD --")


                if(self.just_data_collection):
                    out_dir = self.data_dump_dir+"dynamic/episodes/"+self.cur_episode_id+"/"
                else:
                    out_dir = self.data_dump_dir+"surface/episodes/"+self.cur_episode_id+"/"

                if(os.path.exists(os.path.expanduser(out_dir))):
                    rospy.loginfo("\t Re-using episode dir")
                else:
                    rospy.loginfo("\t Creating episode dir")
                    os.makedirs(os.path.expanduser(out_dir))

                rospy.loginfo("\t Writing")
                img = self.cv_bridge.imgmsg_to_cv2(self.cur_observation_data['rgb_image'])
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                cv2.imwrite(os.path.expanduser(out_dir)+self.cur_scene_id+".png",img)
                data_file = os.path.expanduser(out_dir)+"data.txt"
                if(os.path.isfile(os.path.expanduser(data_file))):
                    rospy.loginfo("\t Data already written")
                else:
                    rospy.loginfo("\t Writing data")
                    with open(data_file,"a+") as f:
                        f.write(str(self.cur_observation_data['waypoint'])+"\n")
                        f.write(str(self.cur_observation_data['timestamp'])+"\n")
                        f.write(str(self.cur_observation_data['metadata'])+"\n")


            else:
                rospy.logerr("couldn't add scene to view store, this is catastrophic")




            return True
        except Exception,e:
            rospy.loginfo(e)
            rospy.loginfo("LEARNING CORE: failed to add view to view store")
            return False


    def process_scene(self,cloud,waypoint,extra_data=None):
        try:
            rospy.loginfo("LEARNING CORE: ---- Storing view in View Store ----")
            self.cur_waypoint = waypoint
            self.populate_observation_data(cloud,extra_data)
            success = self.register_with_view_store(cloud)
            if not success:
                rospy.logerr("Data collection failed for some reason. See above.")
                return ProcessSceneResponse(False,self.cur_view_soma_ids)

            # if we're just doing data collection, no need to do any more processing
            if(self.just_data_collection is True):
                rospy.loginfo("In data collection mode, so not doing any more processing.")
                return ProcessSceneResponse(True,self.cur_view_soma_ids)
                #return


            rospy.loginfo("LEARNING CORE: ---- Segmenting Scene ----")
            self.cur_proc_scene = self.segment_processor.add_unsegmented_scene(self.cur_observation_data,extra_data)
            if(self.cur_proc_scene.clean_setup is True):

                self.cur_proc_scene.waypoint = waypoint

                #if(self.recog_manager):
                #    rospy.loginfo("LEARNING CORE: ---- Running Object Recognition ----")
                #    recognition = self.recog_manager.recognise_scene(cloud)
                #    if(recognition is True):
                #        self.recog_manager.assign_labels(scene)
                #else:
                #    rospy.logwarn("Object recognition service not found, try restarting is the node running?")

                self.assign_segments(self.cur_proc_scene,self.segment_processor.prev_scene,extra_data)
                self.pending_obs.append(self.cur_proc_scene)

                rospy.loginfo("LEARNING CORE: have: " + str(len(self.pending_obs)) + " view(s) waiting to be processed")

                return ProcessSceneResponse(True,self.cur_view_soma_ids)
            else:
                rospy.loginfo("LEARNING CORE: Error in processing scene")
        except Exception,e:
            rospy.logerr("Unable to segment and process this scene -- see the issues above. This may be fine")
            rospy.logerr(e)
            return ProcessSceneResponse(False,self.cur_view_soma_ids)

    def process_scene_callback(self, req):
        rospy.loginfo("-- Received instruction to process a scene")
        result = ProcessSceneResponse(False,self.cur_view_soma_ids)
        if(self.setup_clean is False):
            rospy.logerr("-- surface_based_object_learning node is missing one or more key services, cannot act --")
            rospy.logerr("-- run services and then re-start me --")
            return result
        else:

            if(req.input is None):
                rospy.logwarn("-- This point cloud looks empty, is the service being called correctly? ---")
                rospy.logwarn("-- Stopping Processing ---")
                return result

            self.just_data_collection = req.just_data_collection

            if(self.just_data_collection):
                rospy.loginfo("Running in data collection mode")
            else:
                rospy.loginfo("Running in full process")

            result = self.process_scene(req.input,req.waypoint)

            return result

    def do_postprocessing(self):
        try:
            rospy.loginfo("LEARNING CORE: -- beginning post-processing, attempting view alignment and object label updates -- ")

            if(self.just_data_collection):
                rospy.loginfo("In data collection mode, so just publishing the IDs of the scenes I have observed")
                rospy.loginfo("Publishing " + str(len(self.cur_scene_list)) + " scenes to /surface_based_object_learning/scenes")
                rospy.Rate(10) # 10hz
                for k in self.cur_scene_list:
                    self.scene_publisher.publish(k)
                self.scene_publisher.publish("done")
                return


            if(self.queued_soma_objs):
                rospy.loginfo("-- inserting " + str(len(self.queued_soma_objs)) + " objects into soma")
                ms = SOMANewObjects()
                for k in self.queued_soma_objs:
                    print(k.id)
                    ms.ids.append(k.id)
                self.obj_discovery_publisher.publish(ms)
                res = self.soma_insert(self.queued_soma_objs)
                print(res)


            if(len(self.cur_sequence_obj_ids) == 0):
                rospy.loginfo("LEARNING CORE: -- no segments found in this scene, or if they were they were filtered out by the SOMA region or height filters ")

            for object_id in self.cur_sequence_obj_ids:
                rospy.loginfo("trying to get soma id: " + object_id)
                soma_object = self.get_soma_objects_with_id(object_id)
                rospy.loginfo("trying to get segment:-" + object_id+"-")
                segment = self.get_segment(object_id)
                if(segment.result is False):
                    rospy.logerr("Could not access low-level data for object")
                    rospy.logerr("This is catastrophic")
                    continue
                rospy.loginfo("done")

                observations = segment.response.observations
                rospy.loginfo("LEARNING CORE: observations for " + str(object_id) + " = " + str(len(observations)))
                for o in observations:
                    out_dir = self.data_dump_dir+"surface/episodes/"+self.cur_episode_id+"/objects/"+object_id+"/"
                    img = o.rgb_cropped

                    if(os.path.exists(os.path.expanduser(out_dir))):
                        rospy.loginfo("\t reusing obj dir")
                    else:
                        rospy.loginfo("\t Creating obj dir")
                        os.makedirs(os.path.expanduser(out_dir))

                    img = self.cv_bridge.imgmsg_to_cv2(img)
                    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    outfile = os.path.expanduser(out_dir)+str(observations.index(o))+".png"
                    rospy.loginfo("logging to: " + str(outfile))
                    cv2.imwrite(outfile,rgb_img)

            #    if(len(observations) >= 2):
           #         rospy.loginfo("LEARNING CORE: processing...")
                    # update world model

                    try:
                        rospy.loginfo("LEARNING CORE: updating world model")
                        #merged_cloud = self.view_alignment_manager.register_views(segment.response.observations)
                        rospy.loginfo("LEARNING CORE: updating SOMA obj")
                        soma_object.objects[0].cloud = segment.response.observations[0].map_cloud

                        self.soma_update(object=soma_object.objects[0],db_id=str(object_id))
                    except Exception,e:
                        rospy.logerr("problem updating object models in world/SOMA db.")
                        rospy.logerr(e)
                        continue
                else:
                    rospy.loginfo("LEARNING CORE: not running view alignment, only one view")

          #          try:
         #               rospy.loginfo("LEARNING CORE: updating world model")
        #                merged_cloud = self.view_alignment_manager.register_views(segment.response.observations)
       #                 rospy.loginfo("LEARNING CORE: updating SOMA obj")
      #                  soma_object.objects[0].cloud = merged_cloud

     #                   self.soma_update(object=soma_object.objects[0],db_id=str(object_id))
    #                except Exception,e:
   #                     rospy.logerr("problem updating object models in world/SOMA db. Unable to register merged clouds")
  #                      rospy.logerr(e)
 #                       continue
#                else:
 #                   rospy.loginfo("LEARNING CORE: not running view alignment, only one view")


                #rospy.loginfo("LEARNING CORE: attempting to update object's recognition label")
                #try:
                #    soma_objects.objects[0].type = str(world_object.label)
                #    self.soma_update(object=soma_objects.objects[0],db_id=str(object_id))
                #    rospy.loginfo("LEARNING CORE: done! this object recognised as a " + str(world_object.label) + " with confidence: " + str(world_object.label_confidence))
                #except Exception,e:
                #    rospy.logerr("Problem updating SOMA object label.")
                #    rospy.logerr(e)







            rospy.loginfo("LEARNING CORE: post-processing complete")
        except Exception,e:
            rospy.logerr("Failed at post-processing step")
            rospy.logerr(e)
            rospy.logerr("Abandoning and attempting to clean up")
            self.clean_up_obs()


    def add_soma_object(self,obj):
        rospy.loginfo("LEARNING CORE: getting service")
        rospy.wait_for_service('soma/insert_objects')
        rospy.loginfo("LEARNING CORE: done")
        soma_insert = rospy.ServiceProxy('soma/insert_objects',SOMAInsertObjs)
        soma_insert(obj)

    def get_soma_objects_with_id(self,id):
        rospy.loginfo("LEARNING CORE: looking for SOMA objects with id: " + str(id))
        query = SOMAQueryObjsRequest()
        query.query_type = 0
        query.objectids = ([id])
        #query.objecttypes=['']

        response = self.soma_get(query)


        return response

    def populate_observation_data(self,scene,extra_data=None):
        self.cur_observation_data = {}
        if(extra_data is None):
            rospy.loginfo("LEARNING CORE: *** Making observation using live robot data")
            try:

                        #self.pointcloud_topic = pc
                        #self.rgb_image_topic = rgb
                    #    self.depth_topic = depth
                        #self.tf_topic = tf
                        #self.pose_topic = pose
                    #    self.camera_info_topic = cam
                rospy.loginfo("getting sensor data")
                rospy.loginfo("getting rgb img from " + self.rgb_image_topic)

                self.cur_observation_data['rgb_image'] = rospy.wait_for_message(self.rgb_image_topic, Image, timeout=10.0)

                rospy.loginfo("getting depth img from " + self.depth_image_topic)

                self.cur_observation_data['depth_image'] = rospy.wait_for_message(self.depth_image_topic, Image, timeout=10.0)
                rospy.loginfo("camera_info from " + self.camera_info_topic)

                self.cur_observation_data['camera_info'] = rospy.wait_for_message(self.camera_info_topic, CameraInfo, timeout=10.0)

                self.cur_observation_data['scene_cloud'] = scene
                self.cur_observation_data['waypoint'] = self.cur_waypoint
                self.cur_observation_data['timestamp'] = int(rospy.Time.now().to_sec())
                rospy.loginfo("waiting for robot pose on: " + self.pose_topic)
                self.cur_observation_data['robot_pose'] = rospy.wait_for_message(self.pose_topic, geometry_msgs.msg.Pose, timeout=10.0)
                rospy.loginfo("done")

                meta_data = "{}"
                data_dict = {}
                if(self.just_data_collection):
                    data_dict["source"] = "dynamic_cluster_observations"
                    meta_data = json.dumps(data_dict)
                else:
                    data_dict["source"] = "surface_based_object_learning"
                    meta_data = json.dumps(data_dict)

                self.cur_observation_data['metadata'] = meta_data



                # populates the tf entry with a few seconds worth of tf data
                listener = TransformationStore()
                listener.create_live()
                print("waiting for listener")
                rospy.sleep(2)
                listener.kill()
                self.cur_observation_data['tf'] = listener.get_as_msg()

            except rospy.ROSException, e:
                rospy.logwarn("Failed to get some observation data")
                rospy.logwarn(e)
                return None
        else:
                rospy.loginfo("LEARNING CORE: *** Making observation using historic robot data")
                self.cur_observation_data['rgb_image'] = extra_data['rgb_image']
                self.cur_observation_data['camera_info'] = extra_data['camera_info']
                self.cur_observation_data['scene_cloud'] = extra_data['scene_cloud']
                self.cur_observation_data['robot_pose'] = extra_data['robot_pose']
                self.cur_observation_data['metadata'] = extra_data['metadata']
                self.cur_observation_data['timestamp'] = extra_data['timestamp']
                self.cur_observation_data['tf'] = extra_data['tf']
                self.cur_observation_data['depth_image'] = extra_data['depth_image']
                self.cur_observation_data['waypoint'] = self.cur_waypoint

        return self.cur_observation_data

    def assign_segments(self,scene,prev_scene,extra_data=None):
        rospy.loginfo("LEARNING CORE: Assigning segments")
        cur_scene = scene
        self.cur_view_soma_ids = []
        have_previous_scene = False
        data_dict = {}
        data_dict["source"] = "surface_based_object_learning"
        data_dict["waypoint"] = self.cur_observation_data['waypoint']

        meta_data = json.dumps(data_dict)

        # if this is not scene 0, ie. we have a previous scene to compare to
        if(prev_scene is not None):
            if(prev_scene != scene):
                have_previous_scene = True

        if(have_previous_scene):
            rospy.loginfo("LEARNING CORE: We have a previous scene")
            rospy.loginfo("LEARNING CORE: Current scene ID: " + scene.scene_id)
            rospy.loginfo("LEARNING CORE: Previous scene ID: " + prev_scene.scene_id)

        else:
            rospy.loginfo("LEARNING CORE: we do not have a previous scene")


        if not cur_scene:
            rospy.loginfo("LEARNING CORE: don't have anything in the current scene...")
            rospy.loginfo("LEARNING CORE: did segmentation fail?")
            return

        # we iterate over the INSTANCES OF segmentS visible in the current scene
        # and see if we can link them to previously seen segments in prior views
        for cur_scene_segment_instance in cur_scene.segment_list:

            # this object is a Segment in the database
            # whereas cur_scene_segment_instance is an instance of a segment in memory in the current view
            target_db_segment = None

            # if there are previous views to look at it
            if(have_previous_scene):
                rospy.loginfo("LEARNING CORE: seeing if prev scene contains: " + str(cur_scene_segment_instance.segment_id))
                for pc in prev_scene.segment_list:
                    rospy.loginfo(pc.segment_id)

                if(prev_scene.contains_segment_id(cur_scene_segment_instance.segment_id)):
                    rospy.loginfo("LEARNING CORE: getting EXISTING segment")
                    get_segment_req = self.get_segment(cur_scene_segment_instance.segment_id)
                    if(get_segment_req.result is False):
                        rospy.logerr("Failed to retreive segment, this is catastrophic")
                        return
                    else:
                        target_db_segment = get_segment_req.response

            # if this the first view, or a new segment
            if not target_db_segment:
                rospy.loginfo("LEARNING CORE: creating NEW segment")
                request = self.insert_segment(meta_data,self.cur_scene_id,[])
                if(request.result is False):
                    rospy.logerr("Unable to insert segment, this is catastrophic")
                    return

                cur_scene_segment_instance.segment_id = request.response.id
                target_db_segment = request.response
                self.cur_sequence_obj_ids.append(target_db_segment.id)

            # from here we've either added this as a new object to the scene
            # or retreived the data for it in a previous scene
            if(target_db_segment):
                # so first add a new observation to it, in all cases
                rospy.loginfo("LEARNING CORE: making observation")
                # add an observation for the object

                new_segment_observation = Observation()
                #new_segment_observation.id =  I'm ignoring this because if it's left blank, the service generates one for you
                new_segment_observation.timestamp = self.cur_observation_data['timestamp']
                new_segment_observation.meta_data = meta_data

                new_segment_observation.pose =  cur_scene_segment_instance.map_centroid # centroid in map co-ordinates
                new_segment_observation.map_cloud =  cur_scene_segment_instance.segmented_pc_mapframe #segmented cloud in map co-ordinates
                new_segment_observation.camera_cloud = cur_scene_segment_instance.segmented_pc_camframe # segmented  cloud in camera co-ordinates
                #new_segment_observation.room_cloud = None # segmented cloud aligned to meta-room

                new_segment_observation.rgb_cropped = cur_scene_segment_instance.cropped_rgb_image
                new_segment_observation.depth_cropped =  cur_scene_segment_instance.cropped_depth_image
                new_segment_observation.image_mask = cur_scene_segment_instance.image_mask

                self.append_obs_to_segment(target_db_segment.id,[new_segment_observation],self.cur_scene_id)

                # do some sanity checking
                get_segment_req = self.get_segment(cur_scene_segment_instance.segment_id)
                target_db_segment = get_segment_req.response
                rospy.loginfo("LEARNING CORE: segment: " + target_db_segment.id + " now has " + str(len(target_db_segment.observations)) + " observations")

                cur_soma_obj = None
                soma_objs = self.get_soma_objects_with_id(target_db_segment.id)

                if(soma_objs.objects):
                    rospy.loginfo("soma has this object")
                    cur_soma_obj = soma_objs.objects[0]
                    # nothing to do in this case?

                for k in self.queued_soma_objs:
                    if(k.id in target_db_segment.id):
                        cur_soma_obj = k
                        break

                if(cur_soma_obj):
                    rospy.loginfo("LEARNING CORE: soma has this object")

                else:
                    rospy.loginfo("LEARNING CORE: soma doesn't have this object")
                    # if this object is unknown, lets register a new unknown object in soma
                    #  have a soma object with this id
                    # create it
                    try:
                        cur_soma_obj = SOMAObject()
                        cur_soma_obj.id = target_db_segment.id
                        rospy.loginfo("LEARNING CORE: segment: " + target_db_segment.id + " will have SOMA id " + target_db_segment.id)

                        cur_soma_obj.type = "unknown"
                        #cur_soma_obj.waypoint = self.cur_observation_data['waypoint']



                        cur_soma_obj.metadata = meta_data

                        # either way we want to record this, so just do it here?
                        cur_soma_obj.cloud = new_segment_observation.map_cloud
                        cur_soma_obj.pose = new_segment_observation.pose

                        if(extra_data is not None):
                            cur_soma_obj.logtimestamp = self.cur_observation_data['timestamp'] # jesus christ

                        #cur_soma_obj.sweepCenter = self.cur_observation_data['robot_pose']

                        rospy.loginfo("LEARNING CORE: inserting into SOMA once views are done")

                        self.queued_soma_objs.append(cur_soma_obj)
                        #res = self.soma_insert([cur_soma_obj])


                    except Exception, e:
                        rospy.logerr("unable to insert into SOMA. Is the database server running?")
                        rospy.logerr(e)

                # record all SOMA objects seen in this view
                self.cur_view_soma_ids.append(target_db_segment.id)
                # publish an annotated image of the clusters



                rospy.loginfo("LEARNING CORE: done")
        if(self.cur_proc_scene is not None):

            img = self.cv_bridge.imgmsg_to_cv2(self.cur_observation_data['rgb_image'])

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            cv2.rectangle(img,(0,0),(64,64),(255,0,0),3)

            cv2.imwrite("BASE_SCENE.png",img)

            rospy.loginfo("drawing bounding boxes: ")
            for seg in self.cur_proc_scene.segment_list:
                bx = seg.padded_img_bbox
                rospy.loginfo(bx)
                #int(y_start):int(y_end), int(x_start):int(x_end)
                cv2.rectangle(img,(bx[0],bx[1]),(bx[2],bx[3]),(0,255,0),3)
            rospy.loginfo("trying to write image...")
            rimg = self.cv_bridge.cv2_to_imgmsg(img, encoding="passthrough")
            self.segment_bbox_publisher.publish(rimg)
            cv2.imwrite("ANNOTATED_SEGMENTS.png",img)
            rospy.loginfo("done!")

        rospy.loginfo("LEARNING CORE: DB Update Complete")
        rospy.loginfo("LEARNING CORE: ")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='world_state_manager.py')
    parser.add_argument('surface_roi_tag', nargs=1, help="Tag for filtering SOMa ROIs")
    parser.add_argument("rgb_dump_dir", nargs=1, help='RGB HDD save location')
    parser.add_argument("pointcloud_topic", nargs=1, help='PC TOpic')
    parser.add_argument('rgb_topic', nargs=1, help="RGB Image Topic")
    parser.add_argument("depth_topic", nargs=1, help='Depth Image Topic')
    parser.add_argument('tf_topic', nargs=1, help="TF Topic")
    parser.add_argument('pose_topic', nargs=1, help="Robot Pose Topic")
    parser.add_argument('camera_info_topic', nargs=1, help="Camera Info Topic")


    args = parser.parse_args(rospy.myargv(argv=sys.argv)[1:])

    if(len(sys.argv) < 8):
        rospy.loginfo("LEARNING CORE: not enough args, need everything set!")
    else:
        roitag = str(vars(args)['surface_roi_tag'][0])
        rospy.loginfo("DATA DIR IS:"+roitag)
        datadir = str(vars(args)['rgb_dump_dir'][0])
        rospy.loginfo("ROITAG IS:"+datadir)

		#$(arg surface_roi_tag)
		#$(arg rgb_dump_dir)
		#$(arg pointcloud_topic)
		#$(arg rgb_topic)
	    #$(arg depth_topic)
		#$(arg tf_topic)
		#$(arg pose_topic)
		#$(arg camera_info_topic)

        pc = str(vars(args)['pointcloud_topic'][0])
        depth = str(vars(args)['depth_topic'][0])
        rgb = str(vars(args)['rgb_topic'][0])
        tf = str(vars(args)['tf_topic'][0])
        pose = str(vars(args)['pose_topic'][0])
        cam = str(vars(args)['camera_info_topic'][0])

        #rospy.loginfo("LEARNING CORE: got db_hostname as: " + hostname + " got db_port as: " + port)
        core = LearningCore(roitag,datadir,pc,depth,rgb,tf,pose,cam)
        core.begin_spinning()
