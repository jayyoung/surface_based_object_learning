import roslib
import rospy
from sensor_msgs.msg import PointCloud2, PointField
from world_modeling.srv import *
from soma_io.observation import Observation, TransformationStore
from soma_io.geometry import *
from soma_io.state import World, Object
from soma_io.observation import *
# SOMA2 stuff
from soma2_msgs.msg import SOMA2Object
from soma_manager.srv import *
from geometry_msgs.msg import Pose, Transform, Vector3, Quaternion
import sensor_msgs.point_cloud2 as pc2
import python_pcd
import tf
# reg stuff #
from observation_registration_services.srv import *
from std_srvs.srv import Trigger


if __name__ == '__main__':
    rospy.init_node('BATCH_TEST', anonymous=False)
    world_update = rospy.ServiceProxy('update_world_model', WorldUpdate)
    begin_obs = rospy.ServiceProxy('/begin_observations', Trigger)
    end_obs = rospy.ServiceProxy('/end_observations', Trigger)

    print("sending begin signal")
    begin_obs()
    views = 0
    while(views < 3):

        invar = raw_input('press key to take view')
        print("waiting for pointcloud message")

        cloud = rospy.wait_for_message(
            "/head_xtion/depth_registered/points", PointCloud2)
        print("got a pointcloud, calling service")
        response = world_update(input=cloud, waypoint="WayPoint32")
        print("done")
        print("service response: ")
        print(response)
        views += 1

    print("sending end signal")
    end_obs()
    print("done")
    # rospy.spin()