from setuptools import setup, find_packages

package_name = 'f1tenth_research'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='root@todo.todo',
    description='F1Tenth Research',
    license='TODO',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'teleop = f1tenth_research.teleop_node:main',
            'pure_pursuit = f1tenth_research.pure_pursuit:main',
            'recorder = f1tenth_research.recorder_node:main',
            'il_driver = f1tenth_research.il_driver:main',
            'rma_deployment = f1tenth_research.gazebo_deployment.rma_deployment_node:main',
            'behavior_monitor = f1tenth_research.gazebo_deployment.behavior_monitor:main',
        ],
    },
)
