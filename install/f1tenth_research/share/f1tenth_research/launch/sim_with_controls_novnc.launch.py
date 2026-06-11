from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

import os


def _launch_setup(context, *args, **kwargs):
    actions = []

    sim_launch = os.path.join(
        get_package_share_directory('f1tenth_gym_ros'),
        'launch',
        'gym_bridge_launch.py',
    )
    actions.append(
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(sim_launch)
        )
    )

    actions.append(
        Node(
            package='f1tenth_research',
            executable='teleop',
            name='keyboard_teleop',
            output='screen',
        )
    )

    if LaunchConfiguration('launch_novnc').perform(context).lower() in ('true', '1', 'yes'):
        compose_file = LaunchConfiguration('novnc_compose_file').perform(context)
        novnc_command = (
            f'docker compose -f "{compose_file}" up -d novnc '
            f'|| docker-compose -f "{compose_file}" up -d novnc'
        )
        actions.append(
            ExecuteProcess(
                cmd=['bash', '-lc', novnc_command],
                output='screen',
            )
        )

    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'launch_novnc',
            default_value='true',
            description='Start the noVNC docker-compose service.',
        ),
        DeclareLaunchArgument(
            'novnc_compose_file',
            default_value='/sim_ws/src/f1tenth_gym_ros/docker-compose.yml',
            description='Path to the f1tenth_gym_ros docker-compose file.',
        ),
        OpaqueFunction(function=_launch_setup),
    ])