#!/usr/bin/env python

import roslib
import rospy
import sys
import argparse
from sensor_msgs.msg import PointCloud2, PointField
from cluster_tracker import SOMAClusterTracker
from world_modeling.srv import *
#from geometry_msgs.msg import Pose

# WS stuff
from soma_io.observation import Observation, TransformationStore
from soma_io.geometry import *
from soma_io.state import World, Object
import soma_io.geometry as ws_geom
from sensor_msgs.msg import Image, PointCloud2, CameraInfo, JointState

# SOMA2 stuff
from soma2_msgs.msg import SOMA2Object
from soma_manager.srv import *

# recog stuff
from recognition_srv_definitions.srv import *

# people tracker stuff #
from bayes_people_tracker.msg import PeopleTracker
from upper_body_detector.msg import UpperBodyDetector

talk = True

class WorldStateManager:
    def __init__(self,db_hostname,db_port):
        rospy.init_node('world_state_modeling', anonymous = False)

        if(talk): print("Manager Online")
        # make a cluster tracker
        self.world_model = World(server_host=db_hostname,server_port=int(db_port))
        if(talk): print("world model done")

        self.cluster_tracker = SOMAClusterTracker()

        # TODO: THIS NEEDS TO BE *REGISTERED* EVENTUALLY
        self.transform_store = TransformationStore()
        self.transform_store.create_live()

        # get the current point cloud
        #if(talk): print("waiting for pc")
        #rospy.wait_for_message('/head_xtion/depth_registered/points',PointCloud2)
        #if(talk): print("got it")

        print("setting up services")
        update_world_state = rospy.Service('update_world_model',WorldUpdate,self.object_segment_callback)
        print("world update service running")
        update_person_state = rospy.Service('update_person_model',PersonUpdate,self.person_segment_callback)
        print("person update service running")

        print("setting up SOMA services")
        print("getting insert service")
        rospy.wait_for_service('soma2/insert_objects')
        print("done")
        self.soma_insert = rospy.ServiceProxy('soma2/insert_objects',SOMA2InsertObjs)

        print("getting query service")
        rospy.wait_for_service('soma2/query_db')
        print("done")
        self.soma_get = rospy.ServiceProxy('soma2/query_db',SOMA2QueryObjs)

        print("getting recognition service")
        self.recog_service = rospy.ServiceProxy('/recognition_service/sv_recognition',recognize)

        if(self.recog_service):
            print("recognition service online")
        else:
            print("no recognition service")

        print("-- node setup completed --")


        rospy.spin()

    def person_segment_callback(self,req):
        pid = req.id
        self.assign_people(pid)
        return PersonUpdateResponse(True)

    def assign_people(self,pid):
        print("assigning")
        # do we have a low-level object with this key?
            # if so get it out
            # if not, create it
        cur_person = self.world_model.get_object(pid)

        if(not cur_person):
            print("creating person entry")
            cur_person = self.world_model.create_object(pid)
            cur_person._parent = self.cur_waypoint
        else:
            print("got this person already")


        # record this observation
        DEFAULT_TOPICS = [("/head_xtion/rgb/image_rect_color", Image),
                          ("/head_xtion/depth/image_rect", Image),
                          ("/robot_pose", geometry_msgs.msg.Pose),
                          ("/upper_body_detector/detections", UpperBodyDetector),
                          ("/upper_body_detector/bounding_box_centres", geometry_msgs.msg.PoseArray),
                          ("/bayes_people_tracker/PeopleTracker", PeopleTracker)]



        person_observation = Observation.make_observation(DEFAULT_TOPICS)
        cur_person.add_observation(person_observation)


        people_tracker_output = person_observation.get_message("/bayes_people_tracker/PeopleTracker")
        # get the idx of the thing we want
        person_idx = people_tracker_output.uuids.index(pid)

        soma_objs=self.get_soma_objects_with_id(cur_person.key)
        cur_soma_person = None

        if(soma_objs.objects):
            print("soma has this person")
            # we have a soma object with this id
            # retrieve it
            cur_soma_person = soma_objs.objects[0] # will only ever return 1 anyway, as keys are unique
        else:
            print("soma doesn't have this person")
            # if this object is unknown, lets register a new unknown object in SOMA2
            # we do not have a soma object with this id
            # create it
            cur_soma_person = SOMA2Object()
            cur_soma_person.id = cur_person.key
            cur_soma_person.type = "person"
            cur_soma_person.waypoint = self.cur_waypoint

            # either way we want to record this, so just do it here?
            #cur_soma_person.cloud = cur_scene_cluster.raw_segmented_pc

            cur_soma_person.pose = people_tracker_output.poses[person_idx]

            msg = rospy.wait_for_message("/robot_pose",  geometry_msgs.msg.Pose, timeout=3.0)
            cur_soma_person.sweepCenter = msg

            print("inserting person detection into SOMA")
            res = self.soma_insert([cur_soma_obj])

        # update this object in some way
        # TODO: HOW?


    def object_segment_callback(self, req):

        data = req.input
        self.cur_waypoint = req.waypoint

        print("got data")
        # handles service calls containing point clouds
        self.cur_waypoint = req.waypoint

        if(talk): print("got cloud:" + str(data.header.seq))
        try:
            self.cluster_tracker.add_unsegmented_scene(data)
            self.assign_clusters()
            return WorldUpdateResponse(True)
        except rospy.ServiceException, e:
            if(talk): print "service call failed: %s"%e
            return WorldUpdateResponse(False)



    def cluster_is_live(self,cluster_id):
        if(talk): print("seeing if object exists:" + str(cluster_id) +" in: " + self.cur_waypoint)
        exists = self.world_model.does_object_exist(cluster_id)

        if(exists):
            live_objects = map(lambda x: x.name, self.world_model.get_children(self.cur_waypoint, {'_life_end': None,}))
            if(talk): print("live objects:" + str(live_objects))
            if(talk): print("cid: " + str(cluster_id))
            if(cluster_id in live_objects):
                if(talk): print("it does!")
                return True

        if(talk): print("nope")
        return False

    def cluster_exists(self,cluster_id):
        return self.world_model.does_object_exist(cluster_id)

    def add_soma_object(self,obj):
        print("getting service")
        rospy.wait_for_service('soma2/insert_objects')
        print("done")
        soma_insert = rospy.ServiceProxy('soma2/insert_objects',SOMA2InsertObjs)
        soma_insert(obj)


    def get_soma_objects_with_id(self,id):
        query = SOMA2QueryObjsRequest()

        query.query_type = 0
        query.usetimestep = False
        query.uselowertime =  False
        query.useuppertime =  False
        query.usedates =  False
        query.useweekday =  False
        query.useroi =  False

        query.objectids = ([id])
        query.objecttypes=['']

        response = self.soma_get(query)

        return response

    def assign_clusters(self):
        if(talk): print("assigning")
        cur_scene = self.cluster_tracker.cur_scene
        prev_scene = self.cluster_tracker.prev_scene

        if(talk): print("waiting for insert service")

        rospy.wait_for_service('/message_store/insert')

        if(talk): print("gotcha")

        # if this is not scene 0, ie. we have a previous scene to compare to
        if not prev_scene:
            print("don't have prev")

        if not cur_scene:
            print("don't have cur")
            print("did segmentation fail?")
            return


        for cur_scene_cluster in cur_scene.cluster_list:

            cur_cluster = None

            if(prev_scene):
                if(prev_scene.contains_cluster_id(cur_scene_cluster.cluster_id)):
                    # do we have a living world model for this cluster already?
                    if(self.cluster_is_live(cur_scene_cluster.cluster_id)):
                        #   fetch the world_model for the cluster
                        if(talk): print("got existing object")
                        cur_cluster = self.world_model.get_object(cur_scene_cluster.cluster_id)


            if not cur_cluster:
                if(talk): print("creating object")
                cur_cluster = self.world_model.create_object(cur_scene_cluster.cluster_id)
                cur_cluster._parent = self.cur_waypoint

            # from here we've either added this as a new object to the scene
            # or retreived the data for it in a previous scene
            if(cur_cluster):
                # so first add a new observation to it, in all cases
                if(talk): print("making observation")
                # add an observation for the object

                # TODO: UNHACK THIS TO INCLUDE ROBOT POSE, DOESN'T WORK IN SIM
                DEFAULT_TOPICS = [("/head_xtion/rgb/image_rect_color", Image),
                                  ("/head_xtion/rgb/camera_info", CameraInfo),
                                  ("/head_xtion/depth/points", PointCloud2),
                                  ("/head_xtion/depth_registered/camera_info", CameraInfo),
                                  ("/head_xtion/depth_registered/points", PointCloud2),
                                  ("/head_xtion/depth/camera_info", CameraInfo),
                                  ("/ptu/state", JointState),
                                  ("/robot_pose", geometry_msgs.msg.Pose)]


                cloud_observation = Observation.make_observation(DEFAULT_TOPICS)

                # centroid of this object, in the head_xtion_rgb_optical_frame
                ws_pose = ws_geom.Pose()
                ws_pose.position.x = cur_scene_cluster.map_centroid[0]
                ws_pose.position.y = cur_scene_cluster.map_centroid[1]
                ws_pose.position.z = cur_scene_cluster.map_centroid[2]

                print("observation made")

                #if(talk): print("POSE")
                #if(talk): print(pose.position)
                cur_cluster.add_pose(ws_pose)

            #    print(cur_cluster.identifications)

                # store the segmented point cloud for this cluster
                cloud_observation.add_message(cur_scene_cluster.raw_segmented_pc,"object_cloud")

                # store the cropped rgb image for this cluster
            #    print("result: ")
            #    print(res)
                try:
                    rs = rospy.wait_for_service('/recognition_service/sv_recognition',1)
                    print("recognition online")
                    recog_out = self.seg_service(cur_scene_cluster.raw_segmented_pc)

                    # this should give us back #
                    labels = recog_out.ids
                    confidences = recog_out.confidence
                    # this is all we care about #
                    cloud_observation.recognition = zip(labels,confidences)
                except Exception, e:
                    print("recog not online")

                cur_cluster.add_observation(cloud_observation)

                cur_soma_obj = None

                soma_objs = self.get_soma_objects_with_id(cur_cluster.key)

                if(soma_objs.objects):
                    print("soma has this object")
                    # we have a soma object with this id
                    # retrieve it
                    cur_soma_obj = soma_objs.objects[0]
                else:
                    print("soma doesn't have this object")
                    # if this object is unknown, lets register a new unknown object in SOMA2
                    # we do not have a soma object with this id
                    # create it
                try:
                    cur_soma_obj = SOMA2Object()
                    cur_soma_obj.id = cur_cluster.key
                    cur_soma_obj.type = "unknown"
                    cur_soma_obj.waypoint = self.cur_waypoint

                    # either way we want to record this, so just do it here?
                    cur_soma_obj.cloud = cur_scene_cluster.raw_segmented_pc

                    soma_pose = geometry_msgs.msg.Pose()
                    soma_pose.position.x = cur_scene_cluster.local_centroid[0]
                    soma_pose.position.y = cur_scene_cluster.local_centroid[1]
                    soma_pose.position.z = cur_scene_cluster.local_centroid[2]

                    cur_soma_obj.pose = soma_pose
                    msg = rospy.wait_for_message("/robot_pose",  geometry_msgs.msg.Pose, timeout=3.0)
                    cur_soma_obj.sweepCenter = msg
                    # TODO: everything is unknown for now, but later on we'll change this to a
                    # class or instance distribution
                    print("inserting into SOMA")
                    res = self.soma_insert([cur_soma_obj])
                    #print("result: ")
                    #print(res)
                except Exception, e:
                    print("unable to insert into SOMA. Is the database server running?")


                if(talk): print("done")


        # next we need to clean up the scene, and mark anything no longer observable
        # as not live
        if(prev_scene and cur_scene):
            for prev_scene_cluster in prev_scene.cluster_list:
                # if the cluster observed in the previous scene is not in the current scene
                if not cur_scene.contains_cluster_id(prev_scene_cluster.cluster_id):
                    if(talk): print("cutting object from previous scene")
                    # set the internal model to not live
                    try:
                        prev_cluster = self.world_model.get_object(prev_scene_cluster.cluster_id)
                        prev_cluster.cut()
                    except Exception, e:
                        # we don't even relaly care about this, if it's not in the db
                        # we're actually good to go
                        print("error:")
                        print(e)
                        print("^^^")

            else:
                if(talk): print("object still live, not cutting")

        # do some cleanup in case of crashes or some other errors
        live_objects = map(lambda x: x.name, self.world_model.get_children(self.cur_waypoint, {'_life_end': None,}))
        for o in live_objects:
            if not cur_scene.contains_cluster_id(o):
                if(talk): print("attempting to kill dangling object...")
                try:
                    dangling_obj = self.world_model.get_object(o)
                    dangling_obj.cut()
                    if(talk): print("success")
                except rospy.ServiceException, e:
                    if(talk): print "failed to cut dangling object, is the database server OK?"

            else:
                if(talk): print("got this cluster, not cutting")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='world_state_manager.py')
    parser.add_argument("db_hostname", nargs=1, help='DB Hostname')
    parser.add_argument('db_port', nargs=1, help="DB Port")

    args = parser.parse_args(rospy.myargv(argv=sys.argv)[1:])

    if(len(sys.argv) < 2):
        print("not enough args, need db hostname and port")
    else:
        hostname = str(vars(args)['db_hostname'][0])
        port = str(vars(args)['db_port'][0])

        print("got db_hostname as: " + hostname + " got db_port as: " + port)
        world_state_manager = WorldStateManager(hostname,port)
