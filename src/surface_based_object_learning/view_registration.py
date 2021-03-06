#!/usr/bin/env python


import roslib
import rospy
from sensor_msgs.msg import PointCloud2, PointField
#from world_modeling.srv import *
# SOMA2 stuff
from soma_msgs.msg import SOMAObject
from soma_manager.srv import *
from geometry_msgs.msg import Pose, Transform, Vector3, Quaternion
import sensor_msgs.point_cloud2 as pc2
from soma_llsd_msgs.msg import *
from soma_llsd.srv import *
#import python_pcd
import tf
# reg stuff #
from observation_registration_services.srv import *
import PyKDL
import tf2_ros
from util import TransformationStore

class ViewAlignmentManager:

    def __init__(self):
        #rospy.init_node('world_modeling_view_alignment', anonymous = True)
        rospy.loginfo("VIEW REG: ---created view alignment manager --")
        rospy.loginfo("VIEW REG: waiting for view alignment service additional_view_registration_server from strands_3d_mapping")
        rospy.wait_for_service('/additional_view_registration_server',10)
        self.reg_serv = rospy.ServiceProxy('/additional_view_registration_server',AdditionalViewRegistrationService)
        rospy.loginfo("VIEW REG: got it")


    def transform_to_kdl(self,t):
         return PyKDL.Frame(PyKDL.Rotation.Quaternion(t.rotation.x, t.rotation.y,
                                                      t.rotation.z, t.rotation.w),
                            PyKDL.Vector(t.translation.x,
                                         t.translation.y,
                                         t.translation.z))

    def transform_cloud(self,cloud,translation,rotation):

        tr_s = self.to_transform(translation,rotation)

        t_kdl = self.transform_to_kdl(tr_s)
        points_out = []
        for p_in in pc2.read_points(cloud):
            p_out = t_kdl * PyKDL.Vector(p_in[0], p_in[1], p_in[2])
            points_out.append([p_out[0],p_out[1],p_out[2],p_in[3]])

        fil_fields = []
        for x in cloud.fields:
            if(x.name in "x" or x.name in "y" or x.name in "z" or x.name in "rgb"):
                fil_fields.append(x)

        res = pc2.create_cloud(std_msgs.msg.Header(), fil_fields, points_out)
        return res

    def merge_pcs(self,clouds):
        combined_cloud_points = []
        for c in clouds:
            raw_cloud = pc2.read_points(c)
            int_data = list(raw_cloud)
            for point in int_data:
                x = point[0]
                y = point[1]
                z = point[2]
                colour = point[3]
                combined_cloud_points.append((x,y,z,colour))
        header = std_msgs.msg.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = clouds[0].header.frame_id
        combined_cloud = pc2.create_cloud(header, clouds[0].fields, combined_cloud_points)
        return combined_cloud

    def to_transform(self,translation,rotation):
        tr_s = geometry_msgs.msg.Transform()
        #tr_s.header = std_msgs.msg.Header()
        #tr_s.header.stamp = rospy.Time.now()
        tr_s.translation.x = translation[0]
        tr_s.translation.y = translation[1]
        tr_s.translation.z = translation[2]

        tr_s.rotation.x = rotation[0]
        tr_s.rotation.y = rotation[1]
        tr_s.rotation.z = rotation[2]
        tr_s.rotation.w = rotation[3]
        return tr_s

    def register_current_view_to_waypoint(self,clouds,waypoint,tfs):
        # run something to find out what the meta room xml file is
        # use this as input to reg serv
        clds = []
        tfst = []
        for cloud,tf in zip(clouds,tfs):
            t_st = TransformationStore().msg_to_transformer(tf)
            c_time = t_st.getLatestCommonTime("map",cloud.header.frame_id)
            t,r = t_st.lookupTransform("map",cloud.header.frame_id,c_time)

            cam_trans = geometry_msgs.msg.Transform()
            cam_trans.translation.x = t[0]
            cam_trans.translation.y = t[1]
            cam_trans.translation.z = t[2]

            cam_trans.rotation.x = r[0]
            cam_trans.rotation.y = r[1]
            cam_trans.rotation.z = r[2]
            cam_trans.rotation.w = r[3]
            tfst.append(cam_trans)
            clds.append(cloud)

        response = None
        try:
            response = self.reg_serv(observation_xml=waypoint,additional_views=clds,additional_views_odometry_transforms=tfst)
        except Exception, e:
            rospy.logwarn("Unable to call view registration service")
            #rospy.logwarn(e)

        rospy.loginfo("VIEW REG: done, response: ")
        transform = response.observation_transform
        trans = [transform.translation.x,transform.translation.y,transform.translation.z]
        rot = [transform.rotation.x,transform.rotation.y,transform.rotation.z,transform.rotation.w]
        cl_0 = self.transform_cloud(clds[0],trans,rot)
        cl_1 = self.transform_cloud(clds[1],trans,rot)
        return [cl_0,cl_1]


    def register_scenes(self,cur_scene,prev_scene,root_scene):
        rospy.loginfo("VIEW REG: running transforms")

        cam_cur = self.transform_cloud(cur_scene.unfiltered_cloud,cur_scene.to_map_trans,cur_scene.to_map_rot)
        cam_prev = self.transform_cloud(prev_scene.unfiltered_cloud,prev_scene.to_map_trans,prev_scene.to_map_rot)
        cam_root = self.transform_cloud(root_scene.unfiltered_cloud,root_scene.to_map_trans,root_scene.to_map_rot)

        bases = [root_scene,prev_scene,cur_scene]
        scenes = [cam_root,cam_prev,cam_cur]
        rospy.loginfo("VIEW REG: done")

        map_root_t = self.to_transform(root_scene.to_map_trans,root_scene.to_map_rot)
        map_prev_t = self.to_transform(prev_scene.to_map_trans,prev_scene.to_map_rot)
        map_cur_t = self.to_transform(cur_scene.to_map_trans,cur_scene.to_map_rot)

        transforms = [map_root_t,map_prev_t,map_cur_t]

        # align these clouds
        rospy.loginfo("VIEW REG: calling alignment service")
        rospy.loginfo("VIEW REG: scenes: " + str(len(scenes)))


        response = None
        try:
            response = self.reg_serv(additional_views=scenes,additional_views_odometry_transforms=transforms)
        except Exception, e:
            rospy.logwarn("Unable to call view registration service")
            rospy.logwarn(e)

        rospy.loginfo("VIEW REG: done, response: ")
        print(response)
        view_trans = response.additional_view_transforms
        transformed_clusters = {}

        for ts,scene,orig in zip(view_trans,scenes,bases):
            rot = [ts.rotation.x,ts.rotation.y,ts.rotation.z,ts.rotation.w]
            tls = [ts.translation.x,ts.rotation.y,ts.rotation.z]
            # make new clouds out of the segmented clouds in this scene
            transformed_clusters[orig.scene_id] = []
            for cluster in orig.segment_list:
                # transform data using above transforms, first to cam frame
                cc = self.transform_cloud(cluster.segmented_pc_camframe,orig.to_map_trans,orig.to_map_rot)
                # then to aligned map
                cm = self.transform_cloud(cc,tls,rot)
                transformed_clusters[orig.scene_id].append([cluster.segment_id,cm])

        return transformed_clusters

    def register_views(self,observations,merge_and_write=False):
        seg_clouds = []
        obs_clouds = []
        obs_transforms = []
        time = []
        self.child_camera_frame = "map"
        rospy.loginfo("VIEW REG: beginning view registration ")

        rospy.wait_for_service('/soma_llsd/get_scene')
        self.get_scene = rospy.ServiceProxy('/soma_llsd/get_scene',GetScene)

        for o in observations:
            rospy.loginfo("VIEW REG: getting tf ")

            obs_scene = self.get_scene(o.scene_id)
            tf_p = obs_scene.response.transform
            t_st = TransformationStore().msg_to_transformer(tf_p)
            cam_cloud = o.camera_cloud
            obs_cloud = obs_scene.response.cloud

            self.root_camera_frame = obs_cloud.header.frame_id

            #c_time = t_st.getLatestCommonTime("map",self.root_camera_frame)
            t,r = t_st.lookupTransform("map",self.root_camera_frame,rospy.Time(0))
            cam_trans = geometry_msgs.msg.Transform()
            cam_trans.translation.x = t[0]
            cam_trans.translation.y = t[1]
            cam_trans.translation.z = t[2]

            cam_trans.rotation.x = r[0]
            cam_trans.rotation.y = r[1]
            cam_trans.rotation.z = r[2]
            cam_trans.rotation.w = r[3]

            seg_clouds.append(cam_cloud)
            obs_clouds.append(obs_cloud)

            obs_transforms.append(cam_trans)


        rospy.loginfo("VIEW REG: got: " + str(len(seg_clouds)) + " clouds for object")

        rospy.loginfo("VIEW REG: running service call")
        response = self.reg_serv(additional_views=obs_clouds,additional_views_odometry_transforms=obs_transforms)

        view_trans = response.additional_view_transforms
        rospy.loginfo(view_trans)

        cloud_id = 0
        #transformed_obs_clouds = []
        transformed_seg_clouds = []
        rospy.loginfo("VIEW REG: -- aligning clouds -- ")
        for transform,seg_cloud in zip(view_trans,seg_clouds):
            rot = [transform.rotation.x,transform.rotation.y,transform.rotation.z,transform.rotation.w]
            trs = [transform.translation.x,transform.rotation.y,transform.rotation.z]

            #transformed_obs_cloud = self.transform_cloud(obs_cloud,trs,rot)
            transformed_seg_cloud = self.transform_cloud(seg_cloud,trs,rot)

            #transformed_obs_clouds.append(transformed_obs_cloud)
            transformed_seg_clouds.append(transformed_seg_cloud)

            cloud_id+=1
            #python_pcd.write_pcd("obs"+str(cloud_id)+".pcd", transformed_obs_cloud)
            #python_pcd.write_pcd("seg"+str(cloud_id)+".pcd", transformed_seg_cloud)

            merged_cloud = self.merge_pcs(transformed_seg_clouds)

    #    if(merge_and_write):
        #    rospy.loginfo("VIEW REG: -- merging and writing clouds to files --")
        #    merged_cloud = self.merge_pcs(obs_clouds)
        #    python_pcd.write_pcd("merged_obs_non_aligned.pcd", merged_cloud)

        #    merged_cloud = self.merge_pcs(transformed_obs_clouds)
        #    python_pcd.write_pcd("merged_obs_aligned.pcd", merged_cloud)

        #    merged_cloud = self.merge_pcs(seg_clouds)
        #    python_pcd.write_pcd("merged_seg_non_aligned.pcd", merged_cloud)

        #    python_pcd.write_pcd("merged_seg_aligned.pcd", merged_cloud)

        rospy.loginfo("VIEW REG: success!")


        return merged_cloud

if __name__ == '__main__':
    rospy.loginfo("VIEW REG: hi")
    rospy.init_node('test2', anonymous = False)
    rospy.wait_for_service('/soma_llsd/get_segment')
    get_segment = rospy.ServiceProxy('/soma_llsd/get_segment',GetSegment)
    rospy.loginfo("VIEW REG: got object")
    obj = get_segment("c9a4a17e-fd77-4523-945f-e8a22db76201")

    rospy.loginfo("VIEW REG: observations: " + str(len(obj.response.observations)))
    #pub = rospy.Publisher('/world_modeling/align_and_merge_test', PointCloud2, queue_size=10)

    vr = ViewAlignmentManager()
    merged_cloud = vr.register_views(obj.response.observations)

    rospy.spin()
