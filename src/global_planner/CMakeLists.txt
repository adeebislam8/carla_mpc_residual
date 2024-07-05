cmake_minimum_required(VERSION 3.0.2)
project(global_planner)

find_package(ros_environment REQUIRED)

set(ROS_VERSION $ENV{ROS_VERSION})

if(${ROS_VERSION} EQUAL 1)

  find_package(catkin REQUIRED COMPONENTS 
      rospy 
      roslaunch 
      std_msgs 
      geometry_msgs
      nav_msgs 
      message_generation
      )

  add_message_files(
    FILES
    WorldPose.msg
    FrenetPose.msg
  )
  ## Generate services in the 'srv' folder
  add_service_files(
    FILES
    Frenet2WorldService.srv
    World2FrenetService.srv
  )
  
  catkin_python_setup()

  generate_messages(
    DEPENDENCIES
    std_msgs
    geometry_msgs
    nav_msgs
  )



  if(CATKIN_ENABLE_TESTING)
    roslaunch_add_file_check(launch)
  endif()

  catkin_package(CATKIN_DEPENDS rospy message_runtime)

  catkin_install_python(
    PROGRAMS 
    src/global_path_publisher.py
    src/frenet_world_converter.py
    DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

  install(DIRECTORY launch/
          DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch)

endif()
