import gym
import rospy
import numpy as np
from gym import spaces
import carla
from carla_msgs.msg import CarlaEgoVehicleControl, CarlaEgoVehicleStatus  # pylint: disable=import-error
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import Pose, PoseStamped, PoseWithCovarianceStamped
from std_msgs.msg import Float64
from visualization_msgs.msg import Marker, MarkerArray

class mpcGym(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):

        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        self.map = self.world.get_map()

        random_spawn_point = self.generate_random_spawn_point()
        goal_point = self.generate_random_goal_point(random_spawn_point)
        

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

        # shall i use image/lidar data/ or just the state?
        self.observation_subscriber = rospy.Subscriber(
            '/mpc_controller/observation', ObservationMsg, self.observation_callback, queue_size=1
        )
        
        self.lane_invasion_subscriber = rospy.Subscriber(
            '/carla/ego_vehicle/lane_invasion', Float64, self.lane_invasion_callback, queue_size=1
        )

        self.collision_subscriber = rospy.Subscriber(
            '/carla/ego_vehicle/collision', MarkerArray, self.collision_callback, queue_size=1
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


        self.current_observation = None
        self.new_observation_received = False

        spawn_pose = self.carla_spawn_to_ros_pose(random_spawn_point)
        self.initial_pose_publisher.publish(spawn_pose)

        goal_pose = self.carla_goal_to_ros_pose(goal_point)
        self.goal_publisher.publish(goal_pose)



    def calculate_reward(self):
        NotImplemented

    def generate_random_spawn_point(self):
        spawn_points = self.map.get_spawn_points()
        return np.random.choice(spawn_points)
    
    def generate_random_goal_point(self, random_spawn_point):
        spawn_points = self.map.get_spawn_points()
        goal_point = np.random.choice(spawn_points)
        while random_spawn_point.location.distance(goal_point.location) < 10:
            goal_point = np.random.choice(spawn_points)
            rospy.loginfo("Random goal point is too close to the spawn point. Randomizing goal point again.")

        return goal_point
    def carla_spawn_to_ros_pose(self, carla_pose):
        ros_pose = PoseWithCovarianceStamped()
        ros_pose.header.frame_id = "map"
        ros_pose.position.x = carla_pose.location.x
        ros_pose.position.y = carla_pose.location.y
        ros_pose.position.z = carla_pose.location.z
        ros_pose.orientation.x = carla_pose.rotation.x
        ros_pose.orientation.y = carla_pose.rotation.y
        ros_pose.orientation.z = carla_pose.rotation.z
        ros_pose.orientation.w = carla_pose.rotation.w
        return ros_pose
    
    def carla_goal_to_ros_pose(self, carla_pose):
        ros_pose = PoseStamped()
        ros_pose.header.frame_id = "map"
        ros_pose.pose.position.x = carla_pose.location.x
        ros_pose.pose.position.y = carla_pose.location.y
        ros_pose.pose.position.z = carla_pose.location.z
        ros_pose.pose.orientation.x = carla_pose.rotation.x
        ros_pose.pose.orientation.y = carla_pose.rotation.y
        ros_pose.pose.orientation.z = carla_pose.rotation.z
        ros_pose.pose.orientation.w = carla_pose.rotation.w
        return ros_pose
    
    def observation_callback(self, msg):
        self.current_observation = msg
        self.new_observation_received = True

    def step(self, action):
        # Publish the action
        control_msg = CarlaEgoVehicleControl()
        control_msg.throttle = action[0]
        control_msg.steer = action[1]
        self.action_publisher.publish(control_msg)

        # Wait for the next observation
        self.new_observation_received = False
        while not self.new_observation_received:
            rospy.sleep(0.01)

        # Return the observation
        return self.current_observation, self.calculate_reward(), self.check_done, {}

    def reset(self):
        self.new_observation_received = False
        new_initial_pose = self.world
        # sent a random initial pose and goal
        while not self.new_observation_received:
            rospy.sleep(0.01)
        return self.current_observation


def main(args=None):
    """

    main function

    :return:
    """
    rospy.init("mpc_gym", args=args)

    mpc_gym = None
    # update_timer = None
    try:
        mpc_gym_env = mpcGym()
        # rospy.on_shutdown(mpc_gym.emergency_stop)

        # update_timer = mpc_gym.new_timer(
        #     mpc_gym.control_time_step, lambda timer_event=None: mpc_gym.run_step())

        mpc_gym.spin()

    except KeyboardInterrupt:
        pass

    finally:
        rospy.loginfo('mpc gym shutting down.')
        rospy.shutdown()

if __name__ == "__main__":
    main()
