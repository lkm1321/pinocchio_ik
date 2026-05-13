from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'pinocchio_ik'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*'))        
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='brian',
    maintainer_email='brian@erl-brian.ucsd.edu',
    description='TODO: Package description',
    license='Apache License 2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'velocity_ik = pinocchio_ik.velocity_ik:main',
            'distance_cbf = pinocchio_ik.cbf_node:main',
        ],
    },
)
