from setuptools import setup
import os
from glob import glob

package_name = 'bot'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.py')),
        (os.path.join('share', package_name, 'world'),
            glob('world/*')),
        (os.path.join('share', package_name, 'models', 'turtlebot3_burger_pose'),
            glob('models/turtlebot3_burger_pose/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='studenti',
    maintainer_email='studenti@todo.todo',
    description='Robot RL package',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'train = bot.train:main',
        ],
    },
)