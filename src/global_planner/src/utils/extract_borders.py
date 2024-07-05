#!/usr/bin/env python

# this scripts:
# 1. Set-ups a spectator following the vehicle
# 2. Calculate the border of the road ahead and behind the vehicle
# 3. Draw the border on road 

# can import this class to extract border for global waypoints
import numpy as np
import carla
from agents.navigation.global_route_planner import GlobalRoutePlanner

class ExtractBorders:
    def __init__(self, world, vehicle):
        self.world = world
        self.vehicle = vehicle
        self.spectator = world.get_spectator()
        self.vehicle_pose = vehicle.get_transform()
        self.map = world.get_map()
        self.actors_to_destroy = []

    def draw_waypoints(self, waypoints):
        for waypoint in waypoints:
            self.world.debug.draw_point(waypoint.transform.location, size=0.1, color=carla.Color(155, 10, 20))

   

    def get_global_route(self):
        # get the global route of type Waypoint
        x_random, y_random = np.random.rand(), np.random.rand()
        gr = GlobalRoutePlanner(self.map, 2.0)
        start_waypoint = self.map.get_waypoint(self.vehicle_pose.location,project_to_road=True, lane_type=(carla.LaneType.Driving))
        end_waypoint = self.map.get_waypoint(carla.Location(x=x_random, y=y_random, z=0),project_to_road=True, lane_type=(carla.LaneType.Driving))
        route = gr.trace_route(start_waypoint.transform.location, end_waypoint.transform.location)
        # global_wp = [x[0] for x in route]    ## LIST OF WAYPOINTS

        # return global_wp
        return route
        
    def set_specator(self):

        self.spectator.set_transform(carla.Transform(self.vehicle.get_transform().location + carla.Location(z=50), carla.Rotation(pitch=-90)))
        # print('spectator set at:', self.spectator.get_transform())

    # get the center waypoints of the road ahead and behind the vehicle by a certain distance
    def get_center_waypoints(self):
        vehicle_pose = self.vehicle.get_transform()
        current_waypoint = self.map.get_waypoint(vehicle_pose.location,project_to_road=True, lane_type=(carla.LaneType.Driving))
        # print("current s: ", current_waypoint.s)
        prev_waypoint_list = []
        next_waypoint_list = []
        for i in range(1, 20):
            prev_waypoint = current_waypoint.previous(i)
            next_waypoint = current_waypoint.next(i)
            if prev_waypoint:
                prev_waypoint_list.append(prev_waypoint[0])
            if next_waypoint:
                next_waypoint_list.append(next_waypoint[0])

        waypoint_list = list(reversed(prev_waypoint_list)) + [current_waypoint] + next_waypoint_list

        for waypoint in waypoint_list:

            self.world.debug.draw_point(waypoint.transform.location, size=0.1, color=carla.Color(255, 0, 0), life_time=0.05)


        return waypoint_list

    # get the border waypoints of type Location
    def get_border_waypoints(self, waypoint_list):     

        left_border_list    = []
        right_border_list   = []
        for waypoint in waypoint_list:
            
            left_border_location = carla.Location(
                x=waypoint.transform.location.x + waypoint.lane_width * np.sin(np.deg2rad(waypoint.transform.rotation.yaw))* 0.5,
                y=waypoint.transform.location.y - waypoint.lane_width * np.cos(np.deg2rad(waypoint.transform.rotation.yaw))* 0.5,
                z=waypoint.transform.location.z
            )
            left_border_list.append(left_border_location)
            # self.world.debug.draw_point(left_border_location, size=0.15, color=carla.Color(0, 255, 0))

            right_border_location = carla.Location(
                x=waypoint.transform.location.x - waypoint.lane_width * np.sin(np.deg2rad(waypoint.transform.rotation.yaw)) * 0.5,
                y=waypoint.transform.location.y + waypoint.lane_width * np.cos(np.deg2rad(waypoint.transform.rotation.yaw)) * 0.5,
                z=waypoint.transform.location.z 
            )
            right_border_list.append(right_border_location)
            # self.world.debug.draw_point(right_border_location, size=0.15, color=carla.Color(0, 255, 0))

        return left_border_list, right_border_list



if __name__ == '__main__':
    client = carla.Client('localhost', 2000)
    client.set_timeout(2.0)
    world = client.get_world()
    actors = world.get_actors()
    # find the vehicle
    vehicle = None
    for actor in actors:
        if 'vehicle' in actor.type_id:
            vehicle = actor
            break
    print(vehicle)
    
    extract_borders = ExtractBorders(world, vehicle)

    route = extract_borders.get_global_route()

    global_wp = [x[0] for x in route]    ## LIST OF WAYPOINTS

    # print(type(global_wp))
    print('global_wp:', global_wp[0])

    extract_borders.draw_waypoints(global_wp)

    
    left_border_wp, right_border_wp = extract_borders.get_border_waypoints(global_wp)

    print('left_border_wp:', left_border_wp[0])
    print('right_border_wp:', right_border_wp[0])
    while True:
 
 
        world.tick()
        extract_borders.set_specator()
        waypoint_list = extract_borders.get_center_waypoints()

        # extract_borders.get_border_waypoints(waypoint_list)
        
