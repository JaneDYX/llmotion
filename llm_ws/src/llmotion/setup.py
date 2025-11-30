from setuptools import setup

package_name = 'llmotion'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Install templates (correct path)
        ('share/' + package_name + '/templates', [
            'llmotion/templates/new.txt',
            'llmotion/templates/basic_demo.txt',
            'llmotion/templates/highway.txt',
            'llmotion/templates/intersection.txt',
            'llmotion/templates/memory_highway.txt',
            'llmotion/templates/memory_inter.txt',
            'llmotion/templates/memory_parking.txt',
            'llmotion/templates/navigation.txt',
            'llmotion/templates/parking.txt',
            'llmotion/templates/hotword_command_ref.json',
            'llmotion/templates/hotword_evaluate_ref.json',
        ]),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='krg',
    maintainer_email='krg@example.com',
    description='LLM Motion Controller for ROS2',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'llm_node = llmotion.llm_node:main',
            'keyboard_node = llmotion.keyboard_node:main',
        ],
    },
)
