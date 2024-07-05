#!/usr/bin/env python
## To-do later: In the case of adding global path on top of center line
import rospy
import sys
import threading
import numpy as np
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import ros_compatibility as roscomp
from ros_compatibility.exceptions import *
from ros_compatibility.node import CompatibleNode
from ros_compatibility.qos import QoSProfile, DurabilityPolicy

from nav_msgs.msg import Path, Odometry
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32MultiArray
from scipy.interpolate import CubicSpline
from global_planner.msg import FrenetPose, WorldPose
from global_planner.srv import World2FrenetService, Frenet2WorldService

from tf.transformations import euler_from_quaternion
from utils.frenet_cartesian_converter import FrenetCartesianConverter


# This is responsible for all conversions between the world and frenet frames
    # Needs to be initialized with a global path (done)
    # Provides services for converting between the two frames (done)
    # Publishes the frenet pose of the vehicle at all time (done)
    # Provides service for converting frenet local path to world local path planned by MPC (to-do)
    # To-do later: Visualize the world local paths

class FrenetWorldConverter(CompatibleNode):
    def __init__(self):
        # self._global_path_sub = rospy.Subscriber('/global_path', Path, self._global_path_callback, queue_size=1)
        # self._odometry_sub = rospy.Subscriber(
        #     '/odometry', 
        #     PoseStamped, 
        #     self._odometry_callback, 
        #     queue_size=1)
    
        # self._frenet_pose_pub = rospy.Publisher('/frenet_pose', FrenetPose, queue_size=1)
        super(FrenetWorldConverter, self).__init__("frenet_world_converter")
        role_name = self.get_param("role_name", "ego_vehicle")

        self._global_path_sub = self.new_subscription(
            Path,
            # "/carla/{}/waypoints".format(role_name),
            '/global_planner/{}/waypoints'.format(role_name),
            self._global_path_callback,
            qos_profile=QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        
        self._odometry_sub = self.new_subscription(
            Odometry,
            "/carla/{}/odometry".format(role_name),
            self._odometry_callback,
            qos_profile=10)

        self._frenet_pose_pub = self.new_publisher(
            FrenetPose,
            "/global_planner/{}/frenet_pose".format(role_name),
            qos_profile=10)

        
        self._world2frenet_service = self.new_service(
            World2FrenetService,
            '/world2frenet',
            self._world2frenet_callback)

        self._frenet2world_service = self.new_service(
            Frenet2WorldService,
            '/frenet2world',
            self._frenet2world_callback)

        self._lock = threading.Lock()
        self._global_path_initialized = False
        self._frenet_cartesian_converter = None

    def _global_path_callback(self, msg):
        self._lock.acquire()

        waypoints = []
        for pose in msg.poses:
            waypoints.append([pose.pose.position.x, pose.pose.position.y])
        
        # print("waypoints:", waypoints)
        self.loginfo('Received global path with {} waypoints'.format(len(waypoints)))
        self._frenet_cartesian_converter = FrenetCartesianConverter(waypoints)
        # self._publish_frenet_path()
        self._global_path_initialized = True
        self._lock.release()

    def _odometry_callback(self, msg):
        # self.loginfo('Received odometry')

        if not self._global_path_initialized:
            return
        # self.loginfo('Received odometry')
        self._lock.acquire()
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        orientation = msg.pose.pose.orientation
        orientation = [orientation.x, orientation.y, orientation.z, orientation.w]
        roll, pitch, yaw = euler_from_quaternion(orientation)
        # self.loginfo('x: {}, y: {}, yaw: {}'.format(x, y, yaw))
        s, d, alpha = self._frenet_cartesian_converter.get_frenet([x, y, yaw])
        # self.loginfo('s: {}, d: {}, alpha: {}'.format(s, d, alpha)) 
        self._frenet_pose_pub.publish(FrenetPose(s=s, d=d, yaw_s=alpha))
        self._lock.release()

    def _world2frenet_callback(self, req):
        if not self._global_path_initialized:
            return None
        # self.loginfo("sercive test {}" .format(req))
        req = req.world_pose
        self._lock.acquire()
        s, d, alpha = self._frenet_cartesian_converter.get_frenet([req.x, req.y, req.yaw])
        # self.loginfo("SERVICE: s: {}, d: {}, alpha: {}".format(s, d, alpha))
        self._lock.release()
        
        return FrenetPose(s=s, d=d, yaw_s=alpha)
    
    def _frenet2world_callback(self, req):
        if not self._global_path_initialized:
            return None
        req = req.frenet_pose
        self._lock.acquire()
        x, y, yaw = self._frenet_cartesian_converter.get_cartesian([req.s, req.d, req.yaw_s])
        self._lock.release()
        return WorldPose(x=x, y=y, yaw=yaw)

    # def _publish_frenet_path(self):
    #     # need to publish dense s values and kappa(curvature) values
    #     self._lock.acquire()
    #     s = np.linspace(0, self._frenet_cartesian_converter.x_spline.x[-1], 1000)
    #     d = np.zeros(1000)
    #     alpha = np.zeros(1000)
    #     frenet_path = Path()
    #     frenet_path.header.frame_id = 'map'
    #     for i in range(1000):
    #         pose = PoseStamped()
    #         pose.pose.position.x = s[i]
    #         pose.pose.position.y = d[i]
    #         pose.pose.position.z = alpha[i]
    #         frenet_path.poses.append(pose)
    #     self._frenet_path_pub.publish(frenet_path)
    #     self._lock.release()


def main(args=None):
    """
    main function
    """
    roscomp.init('frenet_world_converter', args=args)
    frenet_world_converter = None
    try:
        frenet_world_converter = FrenetWorldConverter()
        frenet_world_converter.spin()

    except KeyboardInterrupt:
        pass

    finally:
        roscomp.loginfo('frenet coverter shutting down.')
        roscomp.shutdown()



if __name__ == "__main__":
    main()
