import roslib
import rospy
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
import pcl
import numpy as np

if __name__ == '__main__':
    rospy.init_node('ws_repeater', anonymous = False)

    cloud = pcl.load("tsc2.pcd")

    print(cloud.size)

    fil = cloud.make_passthrough_filter()
    fil.set_filter_field_name("z")
    fil.set_filter_limits(0, 1.5)
    cloud_filtered = fil.filter()

    print(cloud_filtered.size)

    seg = cloud_filtered.make_segmenter_normals(ksearch=50)
    seg.set_optimize_coefficients(True)
    seg.set_model_type(pcl.SACMODEL_NORMAL_PLANE)
    seg.set_normal_distance_weight(0.1)
    seg.set_method_type(pcl.SAC_RANSAC)
    seg.set_max_iterations(100)
    seg.set_distance_threshold(0.03)
    indices, model = seg.segment()

    print(model)

    cloud_plane = cloud_filtered.extract(indices, negative=False)
    cloud_plane.to_file("tsc2seg.pcd")

    cloud_cyl = cloud_filtered.extract(indices, negative=True)

    seg = cloud_cyl.make_segmenter_normals(ksearch=50)
    seg.set_optimize_coefficients(True)
    seg.set_model_type(pcl.SACMODEL_CYLINDER)
    seg.set_normal_distance_weight(0.1)
    seg.set_method_type(pcl.SAC_RANSAC)
    seg.set_max_iterations(10000)
    seg.set_distance_threshold(0.05)
    seg.set_radius_limits(0, 0.1)
    indices, model = seg.segment()

    print(model)

    cloud_cylinder = cloud_cyl.extract(indices, negative=False)
    cloud_cylinder.to_file("tsc2clu.pcd")