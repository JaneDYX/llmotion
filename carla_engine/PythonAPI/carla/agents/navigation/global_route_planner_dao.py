#!/usr/bin/env python

import numpy as np
import carla

class GlobalRoutePlannerDAO(object):

    def __init__(self, carla_map, sampling_resolution=2.0):
        self._sampling_resolution = sampling_resolution
        self._carla_map = carla_map

    def get_topology(self):
        return self._carla_map.get_topology()

    def get_waypoint(self, location):
        return self._carla_map.get_waypoint(location, project_to_road=True, lane_type=carla.LaneType.Driving)

    def get_resolution(self):
        return self._sampling_resolution

    def generate_waypoints(self, distance):
        return self._carla_map.generate_waypoints(distance)

    def get_map(self):
        return self._carla_map
