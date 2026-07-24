import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
    bot_pkg = get_package_share_directory('bot')
    turtlebot3_gazebo_pkg = get_package_share_directory('turtlebot3_gazebo')
    turtlebot3_description_pkg = get_package_share_directory('turtlebot3_description')

    world_path = os.path.join(bot_pkg, 'world', 'medium.sdf')

    gz_resource_path = SetEnvironmentVariable(
        name='GZ_SIM_RESOURCE_PATH',
        value=[
            os.path.join(bot_pkg, 'world'),
            ':',
            os.path.join(turtlebot3_gazebo_pkg, 'models')
        ]
    )

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('ros_gz_sim'),
                'launch',
                'gz_sim.launch.py'
            )
        ),
        launch_arguments={
            
            'gz_args': f' -s -r {world_path}'
        }.items()
    )

    tb3_model_path = os.path.join(
        bot_pkg,
        'models',
        'turtlebot3_burger_pose',
        'model.sdf'
    )

    spawn_robot = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-name', 'burger',
            '-file', tb3_model_path,
            '-x', '0.0',
            '-y', '0.0',
            '-z', '0.05'
        ],
        output='screen'
    )

    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            '/scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan',
            '/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist',
            '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
            '/odom@nav_msgs/msg/Odometry[gz.msgs.Odometry',
            '/model/burger/pose@geometry_msgs/msg/Pose[gz.msgs.Pose',
            
        ],
        output='screen'
    )

    urdf_path = os.path.join(
        turtlebot3_description_pkg,
        'urdf',
        'turtlebot3_burger.urdf'
    )

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[
            {
                'robot_description': open(urdf_path).read(),
                'use_sim_time': True,
            }
        ],
        output='screen'
    )

    return LaunchDescription([
        gz_resource_path,
        gazebo,
        spawn_robot,
        bridge,
        robot_state_publisher,
    ])