# -*- coding: utf-8 -*-

"""
Setup for global_planner
"""
import os
from glob import glob
ROS_VERSION = int(os.environ['ROS_VERSION'])

if ROS_VERSION == 1:
    from distutils.core import setup
    from catkin_pkg.python_setup import generate_distutils_setup

    d = generate_distutils_setup(
        packages=['global_planner'],
        package_dir={'': 'src'},
    )

    setup(**d)

elif ROS_VERSION == 2:
    from setuptools import setup

    package_name = 'global_planner'
    setup(
        name=package_name,
        version='0.0.0',
        packages=[package_name],
        data_files=[
            ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
            ('share/' + package_name, ['package.xml']),
            (os.path.join('share', package_name), glob('launch/*.launch.py'))
        ],
        install_requires=['setuptools'],
        zip_safe=True,
        maintainer='Adeeb Islam',
        maintainer_email='adeeb.islam8@gmail.com',
        description='CARLA waypoint publisher for ROS2 bridge',
        license='MIT',
        tests_require=['pytest'],
        entry_points={
            'console_scripts': ['global_planner = global_planner.global_planner:main'],
        },
        package_dir={'': 'src'},
    )
