import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import gym
import rospy
import numpy as np
from gym import spaces
import carla
from carla_msgs.msg import CarlaEgoVehicleControl, CarlaEgoVehicleStatus, CarlaCollisionEvent, CarlaLaneInvasionEvent  # pylint: disable=import-error
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import Pose, PoseStamped, PoseWithCovarianceStamped
from std_msgs.msg import Float64
from visualization_msgs.msg import Marker, MarkerArray
from global_planner.msg import FrenetPose, WorldPose
from global_planner.srv import World2FrenetService, Frenet2WorldService
from src.utils.convert_traj_track import parseReference
from scipy.interpolate import make_interp_spline
from tf.transformations import euler_from_quaternion, quaternion_from_euler
import carla_common.transforms
import threading
import random
secure_random = random.SystemRandom()

DEG2RAD = np.pi/180.0
RAD2DEG = 180.0/np.pi
DIST2OBSTACLE = 6.0
PATH_LENGTH = 100.0

class mpcGym(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        ## manually setting the border width.. later set it with border points
        self.nmax = 1.0
        self.nmin = -0.5  
        self.steer_max = 35 * DEG2RAD  
        self.kappa_spline = None
        self.current_s = 0
        self.lookahead_distance = 100
        self.num_of_obs = 3 
        self.collision = False
        self.lane_invasion = False
        self.mpc_control_initialized = False
        self.selected_obstacles_initialized = False
        self.predicted_path_initialized = False
        self.ego_state_initialized = False
        self.frenet_pose_initialized = False
        self.ref_path_initialized = False
        self.current_observation = None
        self.current_ego_state_info = np.zeros(5)
        self.mpc_control = np.zeros(2)

        self.data_lock = threading.Lock()

        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        self.map = self.world.get_map()


        

        self.action_space = spaces.Box(np.array([-1.0, 1.0]), # normalized throttle/brake
                                    np.array([-1.0, 1.0]), dtype=np.float32)  # normalized steer rate
        observation_space_dict = {
        'state': spaces.Box(
            #                dist ptfx ptfy ptrx ptry wpf0x wpf0y wpf1x wpf1y wpf2x wpf2y wpf3x wpf3y wpr0x wpr0y wpr1x wpr1y wpr2x wpr2y wpr3x wpr3y  vtx atx aty
            low  = np.array([-100, -10, -10, -10, -10, -10,  -10,  -10,  -10,  -10,  -10,  -10,  -10,  -10,  -10,  -10,  -10,  -10,  -10,  -10,  -10,   0,  -10, 10]), 
            high = np.array([100,   10,  10,  10,  10,  10,   10,   10,   10,   10,   10,   10,   10,   10,   10,   10,   10,   10,   10,   10,   10,    20, 10, 10]), 
            shape = (24,),
            dtype=np.float64)
        }
        self.observation_space = spaces.Dict(observation_space_dict)

        ## Observation materials
        self.ego_vehicle_status_subscriber = rospy.Subscriber(
            '/carla/ego_vehicle/vehicle_status', CarlaEgoVehicleStatus, self.ego_state_callback, queue_size=1
        )

        self.selected_obstacles_subscriber = rospy.Subscriber(
            '/mpc_controller/ego_vehicle/selected_obstacles', MarkerArray, self.selected_obstacles_callback, queue_size=1
        )            

        self.reference_path_subscriber = rospy.Subscriber(
            '/global_planner/ego_vehicle/waypoints', Path, self.reference_path_callback, queue_size=1
        )
        
        self.predicted_path_subscriber = rospy.Subscriber(
            '/mpc_controller/ego_vehicle/predicted_path', Path, self.predicted_path_callback, queue_size=1
        )
        
        self.mpc_control_cmd_subscriber = rospy.Subscriber(
            '/mpc_controller/ego_vehicle/mpc_control_mpc_cmd', CarlaEgoVehicleControl, self.mpc_control_cmd_callback, queue_size=1
        )
        
        self.frenet_state_subscriber = rospy.Subscriber(
            "/global_planner/ego_vehicle/frenet_pose", FrenetPose, self.frenet_state_callback, queue_size=1
        )
        
        self.lane_invasion_subscriber = rospy.Subscriber(
            '/carla/ego_vehicle/lane_invasion', CarlaLaneInvasionEvent, self.lane_invasion_callback, queue_size=1
        )

        self.collision_subscriber = rospy.Subscriber(
            '/carla/ego_vehicle/collision', CarlaCollisionEvent, self.collision_callback, queue_size=1
        )



        self.action_publisher = rospy.Publisher(
            '/carla/ego_vehicle/vehicle_control_cmd', CarlaEgoVehicleControl, queue_size=1
        )
        self.goal_publisher = rospy.Publisher(
            '/move_base_simple/goal', PoseStamped, queue_size=1
        )
        self.initial_pose_publisher = rospy.Publisher(
            '/initialpose', PoseWithCovarianceStamped, queue_size=1
        )


        self.world2frenet_service = rospy.ServiceProxy(
            '/world2frenet', World2FrenetService
        )
        self.frenet2world_service = rospy.ServiceProxy(
            '/frenet2world', Frenet2WorldService
        )

        rospy.wait_for_service('/world2frenet')
        rospy.wait_for_service('/frenet2world')
        rospy.loginfo("Initialized the MPC Gym environment.")

        self.reset()

    def emergency_stop(self):
        # with self.data_lock:
            throttle = 0
            steer = 0
            brake = 1
            control_msg = CarlaEgoVehicleControl()
            control_msg.throttle = throttle
            control_msg.steer = steer
            control_msg.brake = brake
            self.action_publisher.publish(control_msg)

    def reset_vehicle(self):
        # with self.data_lock:
            speed = self.current_ego_state_info[0]
            while abs(speed) > 0.1:
                self.emergency_stop()
                speed = self.current_ego_state_info[0]
                rospy.sleep(0.1)
    
            # rospy.sleep(2)

            random_spawn_point = self.generate_random_spawn_point()
            spawn_pose = self.carla_spawn_to_ros_pose(random_spawn_point)
            self.initial_pose_publisher.publish(spawn_pose)

            goal_point = self.generate_random_goal_point(random_spawn_point)
            goal_pose = self.carla_goal_to_ros_pose(goal_point)
            self.goal_publisher.publish(goal_pose)

            rospy.loginfo("Resetting the vehicle to a random spawn point and goal point.")
            rospy.sleep(1)



    def distance(self, p1, p2):
        return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)

    def generate_random_spawn_point(self):
        # with self.data_lock:
            spawn_points = self.map.get_spawn_points()
            spawn_point = secure_random.choice(
                    spawn_points) if spawn_points else carla.Transform()
            spawn_point = carla_common.transforms.carla_transform_to_ros_pose(spawn_point)
            return spawn_point
        
    def generate_random_goal_point(self, random_spawn_point):
        # with self.data_lock:
            spawn_points = self.map.get_spawn_points()
            goal_point = secure_random.choice(
                    spawn_points) if spawn_points else carla.Transform()
            goal_point = carla_common.transforms.carla_transform_to_ros_pose(goal_point)

            while self.distance(random_spawn_point.position, goal_point.position) < PATH_LENGTH:
                goal_point = secure_random.choice(
                    spawn_points) if spawn_points else carla.Transform()
                goal_point = carla_common.transforms.carla_transform_to_ros_pose(goal_point)
                rospy.loginfo("Random goal point is too close to the spawn point. Randomizing goal point again.")

            return goal_point
    
    def carla_spawn_to_ros_pose(self, carla_pose):
        # with self.data_lock:
            ros_pose = PoseWithCovarianceStamped()
            ros_pose.header.frame_id = "map"
            ros_pose.pose.pose = carla_pose
            return ros_pose
    
    def carla_goal_to_ros_pose(self, carla_pose):
        # with self.data_lock:
            ros_pose = PoseStamped()
            ros_pose.header.frame_id = "map"
            ros_pose.pose = carla_pose
            return ros_pose
        

    def _get_world_pose(self, frenet_pose):
        # with self.data_lock:
            request = FrenetPose()
            request.s = frenet_pose.s
            request.d = frenet_pose.d
            response = self.frenet2world_service(request)
            return response.pose
    
    def _get_frenet_pose(self, pose):
        # with self.data_lock:
            request = WorldPose()
            request.x = pose.position.x
            request.y = pose.position.y
            _, _, yaw = euler_from_quaternion([pose.orientation.x, pose.orientation.y, 
                                                pose.orientation.z, pose.orientation.w])
            request.yaw = yaw

            response = self.world2frenet_service(request)
            if response is None:
                self.loginfo("Failed to get frenet pose")
                dummy = FrenetPose()
                dummy.s = 0
                dummy.d = 0
                return dummy
            # self.loginfo("Test _get_frenet_pose: {}".format(response))
            return response.frenet_pose

    def ego_state_callback(self, msg):
        with self.data_lock:
            self.current_speed = msg.velocity
            self.current_accel = np.sqrt(msg.acceleration.linear.x**2 + msg.acceleration.linear.y**2) * 3.6
            self.current_throttle = msg.control.throttle
            self.current_brake = msg.control.brake
            self.current_steering = msg.control.steer
            self.current_ego_state_info = np.array([self.current_speed, self.current_accel, self.current_throttle, self.current_brake, self.current_steering])
            self.ego_state_initialized = True

    def frenet_state_callback(self, msg):
        with self.data_lock:
            self.current_s = msg.s
            self.current_d = msg.d
            self.current_alpha = msg.yaw_s
            self.current_frenet_pose = np.array([self.current_s, self.current_d, self.current_alpha])
            self.frenet_pose_initialized = True

    def mpc_control_cmd_callback(self, msg):
        with self.data_lock:
            steer = msg.steer
            throttle = msg.throttle
            brake = msg.brake
            if brake > 0:
                throttle = -brake
            self.mpc_control = np.array([steer, throttle])
            self.mpc_control_initialized = True

    def selected_obstacles_callback(self, msg):
        with self.data_lock:
            selected_obstacles = []
            for marker in msg.markers:
                selected_obstacles.append([marker.pose.position.x, marker.pose.position.y])
            self.selected_obstacles = np.array(selected_obstacles)
            self.selected_obstacles_initialized = True

    def predicted_path_callback(self, path_msg):
        with self.data_lock:
            predicted_path_cartesian = []
            predicted_path_frenet = []
            for pose in path_msg.poses:
                predicted_path_cartesian.append([pose.pose.position.x, pose.pose.position.y])
                frenet_pose = self._get_frenet_pose(pose.pose)
                predicted_path_frenet.append([frenet_pose.s, frenet_pose.d])

            self.predicted_path_cartesian = np.array(predicted_path_cartesian)
            self.predicted_path_frenet = np.array(predicted_path_frenet)

            self.predicted_path_initialized = True

    def reference_path_callback(self, path_msg):
        with self.data_lock:
            rospy.loginfo("Initializing reference path.")
            path_msg.poses = path_msg.poses[::5]
            self.x_ref_spline, self.y_ref_spline, self.path_length, dense_s, _, kappa = parseReference(path_msg)
            # if none in a.any
            if self.x_ref_spline is None or self.y_ref_spline is None or self.path_length is None or dense_s is None or kappa is None:
                rospy.loginfo("Failed to parse the reference path.")
                self.reset()
                return
            print("Path length: ", self.path_length, "Dense_s: ", dense_s[-1])
            self.kappa_spline = make_interp_spline(dense_s, kappa, k=3)
            # self.path_length = dense_s[-1]
            rospy.loginfo("Reference path initialized.")
            self.ref_path_initialized = True

    def collision_callback(self, msg):
        with self.data_lock:
            self.collision = True
            rospy.loginfo("Collision detected.")
            # self.collision_initialized = True

    def lane_invasion_callback(self, msg):
        with self.data_lock:
            if 10 in msg.crossed_lane_markings:
                self.lane_invasion = True
                rospy.loginfo("Lane invasion detected.")
                # self.lane_invasion_initialized = True
    
    def _calculate_reward(self, observation, action):
        # with self.data_lock:
            if self.collision:
                return -100
            if self.lane_invasion:
                return -50
            
            s = observation[2]
            d = observation[3]

            reward = 0
            if abs(action[0]) > 1:
                reward -= 10
            if abs(action[1]) > self.steer_max:
                reward -= 10
            reward += (s/self.path_length) * 100
            reward -= (self.nmax - d) * 0.1 
            reward -= (d - self.nmin) * 0.1

            for obs in self.selected_obstacles:
                if self.distance_to_obs(obs) < DIST2OBSTACLE:
                    reward -= 5

            return reward

    def distance_to_obs(self, obs):
        # ellipsoid distance: scale the d by 0.5
        return np.sqrt((self.current_frenet_pose[0] - obs[0])**2 + ((self.current_frenet_pose[1] - obs[1])*2)**2)

    def print_initialized(self):
        rospy.loginfo(
             "Ref path: {}, Ego state: {}, Frenet pose: {}, MPC control: {}"
             .format(self.ref_path_initialized, self.ego_state_initialized, self.frenet_pose_initialized, self.mpc_control_initialized)
        )
        rospy.loginfo(
             "Selected obstacles: {}, Predicted path: {}, Collision: {}, Lane Invasion: {}"
             .format(self.selected_obstacles_initialized, self.predicted_path_initialized, self.collision, self.lane_invasion)
        )

    def _get_obs(self):
        # with self.data_lock:
            self._reset_obs()
            rospy.sleep(0.1)
            self.selected_obstacles = np.ones((self.num_of_obs, 2)) * -1000

            self.print_initialized()
            while not self.ref_path_initialized or not self.ego_state_initialized or not self.frenet_pose_initialized or \
                not self.selected_obstacles_initialized or not self.predicted_path_initialized:            
                rospy.loginfo("Waiting for all the topics to get data.")
                rospy.sleep(0.01)

            if self.current_s + self.lookahead_distance > self.path_length:
                self.lookahead_distance = self.path_length - self.current_s
        
            s_list = np.linspace(self.current_s, self.current_s + self.lookahead_distance, 20)
            x_ref_points = self.x_ref_spline(s_list)
            y_ref_points = self.y_ref_spline(s_list)
            kappa_points = self.kappa_spline(s_list)
            reference_sampled_points = [val for val in zip(x_ref_points, y_ref_points, kappa_points)]
            
            reference_sampled_points = np.array(reference_sampled_points)
            observation = np.concatenate([self.mpc_control, self.current_frenet_pose, self.selected_obstacles.flatten(), self.current_ego_state_info,  
                                        reference_sampled_points.flatten(), self.predicted_path_frenet.flatten()], axis=0)

            # print("Observation: ", observation)
            self.current_observation = observation
            return observation 

    def _reset_obs(self):
        # with self.data_lock:
            self.collision = False
            self.lane_invasion = False
            self.mpc_control_initialized = False
            self.selected_obstacles_initialized = False
            self.predicted_path_initialized = False
            self.ego_state_initialized = False
            self.frenet_pose_initialized = False
            rospy.loginfo("Resetting the observation.")
        
    def step(self, residual):
        # with self.data_lock:
            throttle = self.current_observation[0] + residual[0]
            steer = self.current_observation[1] + residual[1]

            # Publish the action
            control_msg = CarlaEgoVehicleControl()
            control_msg.throttle = np.clip(throttle, 0, 1)  
            control_msg.steer = np.clip(steer, -self.steer_max, self.steer_max)
            self.action_publisher.publish(control_msg)
            rospy.loginfo("Stepped with throttle: {} and steer: {}".format(throttle, steer))
            
            observation= self._get_obs()
            reward = self._calculate_reward(observation, [throttle, steer])

            # Return the observation
            return observation, reward, self.check_done(), {}

    def check_done(self):
        # with self.data_lock:
            if self.collision: 
                rospy.loginfo("Collision detected.")
                # rospy.sleep(1)
                # self.reset()
                return True
            if self.lane_invasion:
                rospy.loginfo("Lane invasion detected.")
                # rospy.sleep(1)
                # self.reset()
                return True
            if self.current_s >= (self.path_length - 5):
                rospy.loginfo("Reached the end of the path.")
                rospy.loginfo("Path length: {}".format(self.path_length))
                # self.reset()
                return True
            return False
        


    def reset(self):
        # with self.data_lock:
            self._reset_obs()
            self.reset_vehicle()
            observation = self._get_obs()
            # observation = np.zeros(24)

            return observation, {}

    def render(self, mode='human'):
        pass
    
    def close(self):
        pass

def main(args=None):
    try:
        rospy.init_node('mpc_gym_node')
        mpc_gym_env = mpcGym()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    except KeyboardInterrupt:
        rospy.loginfo('Interrupt received, shutting down.')
    finally:
        rospy.loginfo('Shutting down mpc gym node.')

if __name__ == "__main__":
    main()
