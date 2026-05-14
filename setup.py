from setuptools import setup
import os
from glob import glob

package_name = 'pinocchio_ik'

setup(
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*')),
    ],
)
