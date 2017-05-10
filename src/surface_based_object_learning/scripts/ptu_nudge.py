import rospy
import sys
# Brings in the SimpleActionClient
import actionlib
import topological_navigation.msg
import scitos_ptu.msg
from sensor_msgs.msg import JointState

def reset_gaze():
    rospy.loginfo("Trying to reset gaze")
    ptuClient = actionlib.SimpleActionClient('ResetPtu',scitos_ptu.msg.PtuResetAction)
    ptuClient.wait_for_server()

    goal = scitos_ptu.msg.PtuResetGoal()
   # goal.pan = 0
   # goal.tilt = 0
    #goal.pan_vel = 20
    #goal.tilt_vel = 20
    ptuClient.send_goal(goal)
    ptuClient.wait_for_result()

def look_at_table():
    rospy.loginfo("Trying to look at table")
    ptuClient = actionlib.SimpleActionClient('SetPTUState',scitos_ptu.msg.PtuGotoAction)
    ptuClient.wait_for_server()

    goal = scitos_ptu.msg.PtuGotoGoal()
    goal.tilt = 28 # 30 seems best
    goal.tilt_vel = 5
    goal.pan = 9
    goal.pan_vel = 5
    ptuClient.send_goal(goal)
    ptuClient.wait_for_result()

if __name__ == '__main__':
    rospy.init_node('nudge_ptu', anonymous = True)
    reset_gaze()
    look_at_table()
#    rospy.spin()
