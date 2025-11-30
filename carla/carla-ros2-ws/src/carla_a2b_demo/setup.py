from setuptools import setup

package_name = 'carla_a2b_demo'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='you',
    maintainer_email='you@example.com',
    description='CARLA A2B demo with global planner, local reference and controllers',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'reference_path = carla_a2b_demo.reference_path:main',
            'pure_pursuit = carla_a2b_demo.pure_pursuit:main',
            'stanley_controller = carla_a2b_demo.stanley_controller:main',
            'global_planner = carla_a2b_demo.global_planner:main',
            'local_reference = carla_a2b_demo.local_reference:main',
        ],
    },
)
