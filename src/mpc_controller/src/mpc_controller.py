#!/usr/bin/env python
#
# Copyright (c) 2018-2020 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
#
"""
Todo:
    - Extract vehicle state information and print (done)
    - Convert the global path into frenet frame 
    - Convert the pose data into frenet frame
    - Pass vehicle state information to the MPC controller
    - Implement the border_cb function
    - Implement the border publishing node

"""
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import collections
import math
import threading

import numpy as np
import ros_compatibility as roscomp
from ros_compatibility.node import CompatibleNode
from ros_compatibility.qos import QoSProfile, DurabilityPolicy
from tf.transformations import euler_from_quaternion, quaternion_from_euler

# from carla_ad_agent.vehicle_mpc_controller import VehicleMPCController
from carla_ad_agent.misc import distance_vehicle

from carla_msgs.msg import CarlaEgoVehicleControl, CarlaEgoVehicleStatus  # pylint: disable=import-error
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import Pose, PoseStamped
from std_msgs.msg import Float64
from visualization_msgs.msg import Marker

from global_planner.srv import Frenet2WorldService, World2FrenetService
from global_planner.msg import FrenetPose, WorldPose
from acados_mpc.acados_settings import acados_settings
from utils.convert_traj_track import parseReference
from scipy.interpolate import make_interp_spline

class LocalPlannerMPC(CompatibleNode):
    """
    LocalPlanner implements the basic behavior of following a trajectory of waypoints that is
    generated on-the-fly. The low-level motion of the vehicle is computed by using two PID
    controllers, one is used for the lateral control and the other for the longitudinal
    control (cruise speed).

    When multiple paths are available (intersections) this local planner makes a random choice.
    """

    # minimum distance to target waypoint as a percentage (e.g. within 90% of
    # total distance)
    MIN_DISTANCE_PERCENTAGE = 0.9

    def __init__(self):
        super(LocalPlannerMPC, self).__init__("local_planner_mpc")

        role_name = self.get_param("role_name", "ego_vehicle")
        self.control_time_step = self.get_param("control_time_step", 0.05)

        # Fetch the Q and R matrices from parameters
        self.Q_matrix = self.get_param('~Q_matrix', [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])  # Default Q matrix if not set
        self.R_matrix = self.get_param('~R_matrix', [[1.0, 0.0], [0.0, 1.0]])  # Default R matrix if not set
        self.Tf = 1.5
        self.N = 30
        self.spline_degree = 3
        self.s_list = np.ones(self.N+1)
        # Log the matrices for verification
        # self.loginfo("Q matrix: %s", str(self.Q_matrix))
        # self.loginfo("R matrix: %s", str(self.R_matrix))
        self.data_lock = threading.Lock()

        self._current_pose = None
        self._current_speed = None
        self._current_velocity = None
        self._target_speed = 120.0       # kph
        self._current_accel = None
        self._current_throttle = None
        self._current_brake = None
        self._current_steering = None

        self._buffer_size = 5
        self._waypoints_queue = collections.deque(maxlen=20000)
        self._waypoint_buffer = collections.deque(maxlen=self._buffer_size)
        self._global_path_length = None

        self.acados_solver = None
        self.path_initialized = False
        self._path_msg = None
        # subscribers
        self._odometry_subscriber = self.new_subscription(
            Odometry,
            "/carla/{}/odometry".format(role_name),
            self.odometry_cb,
            qos_profile=10)
        self._ego_status_subscriber = self.new_subscription(
            CarlaEgoVehicleStatus,
            "/carla/{}/vehicle_status".format(role_name),
            self.ego_status_cb,
            qos_profile=10)
        self._path_subscriber = self.new_subscription(
            Path,
            # "/carla/{}/waypoints".format(role_name),
            "/global_planner/{}/waypoints".format(role_name),
            self.path_cb,
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        
        self._world2frenet_service = self.new_client(
            World2FrenetService,
            '/world2frenet')
        self._frenet2world_service = self.new_client(
            Frenet2WorldService,
            '/frenet2world')      

        ## Todo: Implement later ##
        self._border_subscriber = self.new_subscription(
            Path,
            "/carla/{}/border_waypoints".format(role_name),
            self.border_cb,
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        ## Todo: Implement later ##


        self._target_speed_subscriber = self.new_subscription(
            Float64,
            "/carla/{}/speed_command".format(role_name),
            self.target_speed_cb,
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))




        # publishers
        self._target_pose_publisher = self.new_publisher(
            Marker,
            "/mpc_controller/{}/next_target".format(role_name),
            qos_profile=10)
        self._control_cmd_publisher = self.new_publisher(
            CarlaEgoVehicleControl,
            "/carla/{}/vehicle_control_cmd".format(role_name),
            qos_profile=10)
        
        self._reference_path_publisher = self.new_publisher(
            Path,
            "/mpc_controller/{}/reference_path".format(role_name),
            qos_profile=10)

        self._predicted_path_publisher = self.new_publisher(
            Path,
            "/mpc_controller/{}/predicted_path".format(role_name),
            qos_profile=10)
        # initializing controller
        # self._vehicle_controller = VehicleMPCController(
        #     self)

    def odometry_cb(self, odometry_msg):
        # self.loginfo("Received odometry message")
        with self.data_lock:
            self._current_pose = odometry_msg.pose.pose
            # self.loginfo("odom callback: {}".format(self._current_pose))
            self._current_speed = math.sqrt(odometry_msg.twist.twist.linear.x ** 2 +
                                            odometry_msg.twist.twist.linear.y ** 2 +
                                            odometry_msg.twist.twist.linear.z ** 2) * 3.6 # m/s to km/h
            self._draw_reference_point(self._current_pose)

    def _draw_reference_point(self, pose):
        ref_path = Path()
        ref_path.header.frame_id = "map"
        ref_path.header.stamp = roscomp.ros_timestamp(self.get_time(), from_sec=True)

        # print("Current pose: ", pose)
        frenet_pose = self._get_frenet_pose(pose)
        # self.loginfo("Frenet pose: {}".format(frenet_pose))
        s, d = frenet_pose.s, frenet_pose.d
        # 10 waypoints 10m ahead of the vehicle: 
            # todo: make sure the waypoints are within the length of the path
        for i in range(10):
            request = FrenetPose()
            request.s = s + i * 0.5
            request.d = 0
            request.yaw_s = 0
            response = self._frenet2world_service(request)
            pose_msg = self._world2pose(response)
            pose_stamped = PoseStamped()
            pose_stamped.pose = pose_msg
            ref_path.poses.append(pose_stamped)

        self._reference_path_publisher.publish(ref_path)
    # converts WorldPose to geometry_msgs/Pose
    def _world2pose(self, world_pose):
        world_pose = world_pose.world_pose
        pose = Pose()
        pose.position.x = world_pose.x
        pose.position.y = world_pose.y
        pose.position.z = 0
        yaw = world_pose.yaw
        # self.loginfo("Test _worl2pose Yaw: {}".format(yaw))
        pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = quaternion_from_euler(0, 0, yaw)
        # pose.orientation.x = 0
        # pose.orientation.y = 0
        # pose.orientation.z = 0
        # pose.orientation.w = 1
        return pose

    def _get_world_pose(self, frenet_pose):
        request = FrenetPose()
        request.s = frenet_pose.s
        request.d = frenet_pose.d
        response = self._frenet2world_service(request)
        return response.pose
    
    def _get_frenet_pose(self, pose):
        request = WorldPose()
        request.x = pose.position.x
        request.y = pose.position.y
        _, _, yaw = euler_from_quaternion([pose.orientation.x, pose.orientation.y, 
                                            pose.orientation.z, pose.orientation.w])
        request.yaw = yaw
        # request.v = self._current_speed
        # request.acc = self._current_accel
        # request.target_v = self._target_speed

        response = self._world2frenet_service(request)
        if response is None:
            self.loginfo("Failed to get frenet pose")
            dummy = FrenetPose()
            dummy.s = 0
            dummy.d = 0
            return dummy
        # self.loginfo("Test _get_frenet_pose: {}".format(response))
        return response.frenet_pose

    def ego_status_cb(self, ego_status_msg):
        with self.data_lock:
            self._current_accel = math.sqrt(ego_status_msg.acceleration.linear.x ** 2 +
                                            ego_status_msg.acceleration.linear.y ** 2 +
                                            ego_status_msg.acceleration.linear.z ** 2) * 3.6
            self._current_throttle = ego_status_msg.control.throttle
            self._current_brake = ego_status_msg.control.brake
            self._current_steering = ego_status_msg.control.steer
            self._current_velocity = ego_status_msg.velocity
        ## Todo: Check if 3.6 is the correct conversion factor ##         



    def target_speed_cb(self, target_speed_msg):
        with self.data_lock:
            self._target_speed = target_speed_msg.data

    def path_cb(self, path_msg):
        with self.data_lock:
            self._waypoint_buffer.clear()
            self._waypoints_queue.clear()
            self._waypoints_queue.extend([pose.pose for pose in path_msg.poses])
            self._path_msg = path_msg
            # self.loginfo("Received path message of length: {}".format(len(path_msg)))
            # self.loginfo("Current waypoints queue length: {}".format(len(self._waypoints_queue)))
            # self.loginfo("Current waypoints buffer length: {}".format(len(self._waypoint_buffer)))
            # self.loginfo("First waypoint in queue: {}".format(self._waypoints_queue[0]))

            # sparsify path_msg
            path_msg.poses = path_msg.poses[::5]
            _, _, _, dense_s, _, kappa = parseReference(path_msg)
            kappa_spline = make_interp_spline(dense_s, kappa, k=3)
            self.spline_coeffs = kappa_spline.c
            self.spline_knots = kappa_spline.t
            self.loginfo("Spline coefficients: {}".format(self.spline_coeffs))
            self.loginfo("Spline knots: {}".format(self.spline_knots))
            self.path_initialized = True
            self._global_path_length = dense_s[-1]

            self.acados_solver = None
            self.loginfo("Acados Reset")

    ## Todo: Write border publishing node and implement this function ##
    def border_cb(self, path_msg):
        pass
        with self.data_lock:
            self._waypoint_buffer.clear()
            self._waypoints_queue.clear()
            self._waypoints_queue.extend([pose.pose for pose in path_msg.poses])
    ##  -------------------------------------------------------------- ##

    def pose_to_marker_msg(self, pose):
        marker_msg = Marker()
        marker_msg.type = 0
        marker_msg.header.frame_id = "map"
        marker_msg.pose = pose
        marker_msg.scale.x = 1.0
        marker_msg.scale.y = 0.2
        marker_msg.scale.z = 0.2
        marker_msg.color.r = 255.0
        marker_msg.color.a = 1.0
        return marker_msg

    def run_step(self):
        """
        Sets up the OCP problem in acados and solves it
         - initializes the acados 
         - print:
            - current cartesian state (pose+velocity)
            - current frenet state (s, d, yaw_s + frenet velocity, acceleration)
            - current vehicle actuators (throttle, brake, steering)

        """
        with self.data_lock:
        # debug info
            while not self.path_initialized:
                self.loginfo("Waiting for path to be initialized")
                return
            while not self._current_pose:
                self.loginfo("Waiting for odometry message")
                return
            
            # self.loginfo("Current speed: {}".format(self._current_speed))
            # self.loginfo("Current pose: {}".format(self._current_pose)) 
            # self.loginfo("Current velocity: {}".format(self._current_velocity))
            # self.loginfo("Target speed: {}".format(self._target_speed))
            # self.loginfo("Current throttle: {}".format(self._current_throttle))
            # self.loginfo("Current brake: {}".format(self._current_brake))
            # self.loginfo("Current steering: {}".format(self._current_steering))
            # self.loginfo("Current acceleration: {}".format(self._current_accel))
            # self.loginfo("Frenet pose: {}".format(self._get_frenet_pose(self._current_pose)))
            # return
            # initiailize the acados problem
            if self.acados_solver is None:
                self.constraint, self.model, self.acados_solver = acados_settings(self.Tf, self.N, self.spline_coeffs, self.spline_knots, self._path_msg, self.spline_degree)
                self.loginfo("Initialized acados solver")

            # setup ocp
            frenet_pose = self._get_frenet_pose(self._current_pose)
            # self.loginfo("Frenet pose in run step: {}".format(frenet_pose))
            if self._current_brake != 0:
                D = -self._current_brake
            else:
                D = self._current_throttle

            s, n, alpha, v, D, delta = frenet_pose.s, frenet_pose.d, frenet_pose.yaw_s, self._current_speed, D, self._current_steering
            print("s: ", s)
            print("n: ", n) 
            print("global_path_length: ", self._global_path_length)
            # x = [s, n, alpha, v, D, delta]
            self.acados_solver.set(0, "x", np.array([s, n, alpha, v, D, delta]))
            distance2stop = 0.5 * v
            for i in range(1, self.N):
                s_target = s + self._target_speed * (self.Tf / self.N) * (i+1)
                if s_target > self._global_path_length:
                    s_target = self._global_path_length - distance2stop
                # print("s_target_{}: {}".format(i,s_target))

                yref = np.array([
                    s_target,     # s
                    0,                                                       # n
                    0,                                                       # alpha
                    0,                                                       # v
                    0,                                                       # D
                    0,                                                       # delta
                    # self.current_time + (self.Tf/self.N) * (i+1),                                # time
                    0,                                                       # derD   
                    0,                                                       # derdelta
                ])
                self.acados_solver.set(i, "yref", yref)

                self.acados_solver.constraints_set(i, "lh", np.array([
                    self.constraint.along_min,
                    self.constraint.alat_min,
                    self.model.n_min,
                    self.model.v_min,
                    self.model.throttle_min,
                    self.model.delta_min
                ]))
                self.acados_solver.constraints_set(i, "uh", np.array([
                    self.constraint.along_max,
                    self.constraint.alat_max,
                    self.model.n_max,
                    self.model.v_max,
                    self.model.throttle_max,
                    self.model.delta_max
                ]))

            s_target = s + self._target_speed * self.Tf
            if s_target > self._global_path_length:
                s_target = self._global_path_length - distance2stop
            # print("s_target_N: ", s_target)
            yref_N = np.array([
                s_target,     # s
                0,                                     # n
                0,                                     # alpha
                0,                                     # v
                0,                                     # D
                0,                                      # delta
                # 0,                                      # time
            ])
            self.acados_solver.set(self.N, "yref", yref_N)
            self.acados_solver.constraints_set(0, "lbx", np.array([s, n, alpha, v, D, delta]))
            self.acados_solver.constraints_set(0, "ubx", np.array([s, n, alpha, v, D, delta]))
            
            # solve ocp
            status = self.acados_solver.solve()
            if status != 0:
                self.loginfo("acados returned status {}".format(status))
                if status == 1:
                    self.loginfo("solver failed")
                    self.emergency_stop()
                    self.loginfo("Emergency stop")
                    return
                elif status == 2:
                    self.loginfo("Max number of iterations reached")
                elif status == 3:
                    self.loginfo("Minimum step size reached")
                elif status == 4:
                    self.loginfo("QP solver failed")
                    # self.emergency_stop()
                    self.loginfo("Emergency stop")
                    return

            # get solution
            for i in range(self.N + 1):
                x = self.acados_solver.get(i, "x")
                self.s_list[i] = x[0]
                
            solution_list = []
            for i in range(self.N):
                solution_list.append(self.acados_solver.get(i, "x"))

            isNaN = False
            predicted_path = Path()
            predicted_path.header.frame_id = "map"
            predicted_path.header.stamp = roscomp.ros_timestamp(self.get_time(), from_sec=True)

            for i, solution in enumerate(solution_list):
                if np.isnan(solution).any():
                    self.loginfo("Nan in solution at index {}".format(i))
                    isNaN = True
                    break

                # self.loginfo("Solution{}: {}".format(i, solution))
                req = FrenetPose(solution[0], 0, 0, solution[1], 0, 0, 0)
                resp = self._frenet2world_service(req)
                pose_msg = self._world2pose(resp)
                pose_stamped = PoseStamped()
                pose_stamped.pose = pose_msg
                predicted_path.poses.append(pose_stamped)

            # print('          s          n     alpha      v         D     delta')
            for i in range(0, self.N+1, 1):
                x = self.acados_solver.get(i, "x")
                # print(f"x{i}: , {x[0]:8.4f}, {x[1]:8.4f}, {x[2]:8.4f}, {x[3]:8.4f}, {x[4]:8.4f}, {x[5]:8.4f}")

            # self._predicted_path_publisher.publish(predicted_path)

            # draw computed trajectory
            if not isNaN:
                # predicted_path_msg = Path()
                # predicted_path_msg.poses = predicted_path
                self._predicted_path_publisher.publish(predicted_path)

                x0 = self.acados_solver.get(1, "x")
                u0 = self.acados_solver.get(1, "x")
                # self.accel = x0[6]
                # print("self.accel: ", self.accel)

                # self.derD = u0[0]
                # self.derdelta = u0[1]
                # self.jerk = u0[2]
                # print("jerk: ", self.jerk)

                self.target_D = x0[4]
                self.target_delta = x0[5]
                # print("target_D: ", self.target_D)
                # print("target_delta: ", self.target_delta)
                if self.target_D >= 0:
                    self.target_gas = self.target_D
                    self.target_brake = 0

                else:
                    self.target_brake = -self.target_D
                    self.target_gas = 0

                self.target_steer = self.target_delta 

                control_msg = CarlaEgoVehicleControl()
                control_msg.steer = self.target_steer
                control_msg.throttle = self.target_gas
                control_msg.brake = self.target_brake
                control_msg.hand_brake = False
                control_msg.manual_gear_shift = False
                # print("Control message: ", control_msg)
                self._control_cmd_publisher.publish(control_msg)
                # control_msg = Control_Signal()
                # control_msg.gas = self.target_gas
                # control_msg.brake = self.target_brake
                # control_msg.steerangle = self.target_steer
                # self._vehicle_cmd_publisher.publish(control_msg)
                # return

            # 



            
            # if not self._waypoint_buffer and not self._waypoints_queue:
            #     self.loginfo("Waiting for a route...")
            #     self.emergency_stop()
            #     return

            # # when target speed is 0, brake.
            # if self._target_speed == 0.0:
            #     self.emergency_stop()
            #     return

            # #   Buffering the waypoints
            # if not self._waypoint_buffer:
            #     for i in range(self._buffer_size):
            #         if self._waypoints_queue:
            #             self._waypoint_buffer.append(self._waypoints_queue.popleft())
            #         else:
            #             break

            # # target waypoint
            # target_pose = self._waypoint_buffer[0]
            # self._target_pose_publisher.publish(self.pose_to_marker_msg(target_pose))

            # # move using PID controllers
            # control_msg = self._vehicle_controller.run_step(
            #     self._target_speed, self._current_speed, self._current_pose, target_pose)

            # # purge the queue of obsolete waypoints
            # max_index = -1

            # sampling_radius = self._target_speed * 1 / 3.6  # search radius for next waypoints in seconds
            # min_distance = sampling_radius * self.MIN_DISTANCE_PERCENTAGE

            # for i, route_point in enumerate(self._waypoint_buffer):
            #     if distance_vehicle(route_point, self._current_pose.position) < min_distance:
            #         max_index = i
            # if max_index >= 0:
            #     for i in range(max_index + 1):
            #         self._waypoint_buffer.popleft()

            # self._control_cmd_publisher.publish(control_msg)

    def emergency_stop(self):
        control_msg = CarlaEgoVehicleControl()
        control_msg.steer = 0.0
        control_msg.throttle = 0.0
        control_msg.brake = 0.9
        control_msg.hand_brake = False
        control_msg.manual_gear_shift = False
        self._control_cmd_publisher.publish(control_msg)


def main(args=None):
    """

    main function

    :return:
    """
    roscomp.init("local_planner_mpc", args=args)

    local_planner_mpc = None
    update_timer = None
    try:
        local_planner_mpc = LocalPlannerMPC()
        roscomp.on_shutdown(local_planner_mpc.emergency_stop)

        update_timer = local_planner_mpc.new_timer(
            local_planner_mpc.control_time_step, lambda timer_event=None: local_planner_mpc.run_step())

        local_planner_mpc.spin()

    except KeyboardInterrupt:
        pass

    finally:
        roscomp.loginfo('Local planner shutting down.')
        roscomp.shutdown()

if __name__ == "__main__":
    main()
