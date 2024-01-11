import os
import copy

import cv2
import numpy as np
from PIL import Image

from omnigibson.maps.map_base import BaseMap
from omnigibson.utils.ui_utils import create_module_logger
from omnigibson.utils.motion_planning_utils import astar

# Create module logger
log = create_module_logger(module_name=__name__)


class TraversableMap(BaseMap):
    """
    Traversable scene class.
    Contains the functionalities for navigation such as shortest path computation
    """

    def __init__(
        self,
        map_resolution=0.1,
        default_erosion_radius=0.2,
        trav_map_with_objects=True,
        num_waypoints=10,
        waypoint_resolution=0.2,
    ):
        """
        Args:
            map_resolution (float): map resolution in meters, each pixel represents this many meters;
                                    normally, this should be between 0.01 and 0.1
            default_erosion_radius (float): default map erosion radius in meters
            trav_map_with_objects (bool): whether to use objects or not when constructing graph
            num_waypoints (int): number of way points returned
            waypoint_resolution (float): resolution of adjacent way points
        """
        # Set internal values
        self.map_default_resolution = 0.01  # each pixel == 0.01m in the dataset representation
        self.default_erosion_radius = default_erosion_radius
        self.trav_map_with_objects = trav_map_with_objects
        self.num_waypoints = num_waypoints
        self.waypoint_interval = int(waypoint_resolution / map_resolution)

        # Values loaded at runtime
        self.trav_map_original_size = None
        self.trav_map_size = None
        self.mesh_body_id = None
        self.floor_heights = None
        self.floor_map = None

        # Run super method
        super().__init__(map_resolution=map_resolution)

    def _load_map(self, maps_path, floor_heights=(0.0,)):
        """
        Loads the traversability maps for all floors

        Args:
            maps_path (str): Path to the folder containing the traversability maps
            floor_heights (n-array): Height(s) of the floors for this map

        Returns:
            int: Size of the loaded map
        """
        if not os.path.exists(maps_path):
            log.warning("trav map does not exist: {}".format(maps_path))
            return

        self.floor_heights = floor_heights
        self.floor_map = []
        map_size = None
        for floor in range(len(self.floor_heights)):
            if self.trav_map_with_objects:
                # TODO: Shouldn't this be generated dynamically?
                trav_map = np.array(Image.open(os.path.join(maps_path, "floor_trav_{}.png".format(floor))))
            else:
                trav_map = np.array(Image.open(os.path.join(maps_path, "floor_trav_no_obj_{}.png".format(floor))))

            # If we do not initialize the original size of the traversability map, we obtain it from the image
            # Then, we compute the final map size as the factor of scaling (default_resolution/resolution) times the
            # original map size
            if self.trav_map_original_size is None:
                height, width = trav_map.shape
                assert height == width, "trav map is not a square"
                self.trav_map_original_size = height
                map_size = int(
                    self.trav_map_original_size * self.map_default_resolution / self.map_resolution
                )

            # We resize the traversability map to the new size computed before
            trav_map = cv2.resize(trav_map, (map_size, map_size))

            # We make the pixels of the image to be either 0 or 255
            trav_map[trav_map < 255] = 0

            self.floor_map.append(trav_map)

        return map_size

    @property
    def n_floors(self):
        """
        Returns:
            int: Number of floors belonging to this map's associated scene
        """
        return len(self.floor_heights)

    def get_random_point(self, floor=None, prev_point=None, robot=None):
        """
        Sample a random point on the given floor number. If not given, sample a random floor number.
        If @prev_point is given, sample a point in the same connected component as the previous point.

        Args:
            floor (None or int): floor number. None means the floor is randomly sampled
            prev_point (None or 2-tuple): if given, sample a point in the same connected component as the previous point
                - int: floor number of previous point
                - 3-array: (x,y,z) previous point

        Returns:
            2-tuple:
                - int: floor number. This is the sampled floor number if @floor is None
                - 3-array: (x,y,z) randomly sampled point
        """

        # If the given floor and prev_point are not on the same floor, raise an error
        if floor and prev_point and floor != prev_point[0]:
            raise ValueError("floor and prev_point are not on the same floor")
        # If nothing is given, sample a random floor and a random point on that floor
        if floor is None and prev_point is None:
            floor = np.random.randint(0, self.n_floors)
        
        # create a deep copy so that we don't erode the original map
        trav_map = copy.deepcopy(self.floor_map[floor])
        
        # Erode the traversability map to account for the robot's size
        if robot:
            if hasattr(robot, 'tucked_aabb_extent'):
                robot_chassis_extent = robot.tucked_aabb_extent[:2]
            else:
                robot_chassis_extent = robot.aabb_extent[:2]
            robot_radius = np.linalg.norm(robot_chassis_extent) / 2.0
            robot_radius_pixel = int(np.ceil(robot_radius / self.map_resolution))
            trav_map = cv2.erode(trav_map, np.ones((robot_radius_pixel, robot_radius_pixel)))
        else:
            # convert default_erosion_radius from meter to pixel
            erosion_radius_pixel = int(np.ceil(self.default_erosion_radius / self.map_resolution))
            trav_map = cv2.erode(trav_map, np.ones((erosion_radius_pixel, erosion_radius_pixel)))
        
        if prev_point:
            # Find connected component
            _, component_labels = cv2.connectedComponents(trav_map, connectivity=8)

            # If previous point is given, sample a point in the same connected component
            prev_xy_map = self.world_to_map(prev_point[1][:2])
            prev_label = component_labels[prev_xy_map[0]][prev_xy_map[1]]
            trav_space = np.where(component_labels == prev_label)
        else:
            trav_space = np.where(trav_map == 255)
        idx = np.random.randint(0, high=trav_space[0].shape[0])
        xy_map = np.array([trav_space[0][idx], trav_space[1][idx]])
        x, y = self.map_to_world(xy_map)
        z = self.floor_heights[floor]
        return floor, np.array([x, y, z])

    def get_shortest_path(self, floor, source_world, target_world, entire_path=False, robot=None):
        """
        Get the shortest path from one point to another point.
        If any of the given point is not in the graph, add it to the graph and
        create an edge between it to its closest node.

        Args:
            floor (int): floor number
            source_world (2-array): (x,y) 2D source location in world reference frame (metric)
            target_world (2-array): (x,y) 2D target location in world reference frame (metric)
            entire_path (bool): whether to return the entire path
            robot (None or BaseRobot): if given, erode the traversability map to account for the robot's size

        Returns:
            2-tuple:
                - (N, 2) array: array of path waypoints, where N is the number of generated waypoints
                - float: geodesic distance of the path
        """
        source_map = tuple(self.world_to_map(source_world))
        target_map = tuple(self.world_to_map(target_world))

        # create a deep copy so that we don't erode the original map
        trav_map = copy.deepcopy(self.floor_map[floor])

        # erode the traversability map to account for the robot's size
        if robot:
            if hasattr(robot, 'tucked_aabb_extent'):
                robot_chassis_extent = robot.tucked_aabb_extent[:2]
            else:
                robot_chassis_extent = robot.aabb_extent[:2]
            robot_radius = np.linalg.norm(robot_chassis_extent) / 2.0
            robot_radius_pixel = int(np.ceil(robot_radius / self.map_resolution))
            trav_map = cv2.erode(trav_map, np.ones((robot_radius_pixel, robot_radius_pixel)))
        else:
            # convert default_erosion_radius from meter to pixel
            erosion_radius_pixel = int(np.ceil(self.default_erosion_radius / self.map_resolution))
            trav_map = cv2.erode(trav_map, np.ones((erosion_radius_pixel, erosion_radius_pixel)))

        path_map = astar(trav_map, source_map, target_map)
        if path_map is None:
            raise ValueError("No path found")
        path_world = self.map_to_world(path_map)
        geodesic_distance = np.sum(np.linalg.norm(path_world[1:] - path_world[:-1], axis=1))
        path_world = path_world[:: self.waypoint_interval]

        if not entire_path:
            path_world = path_world[: self.num_waypoints]
            num_remaining_waypoints = self.num_waypoints - path_world.shape[0]
            if num_remaining_waypoints > 0:
                remaining_waypoints = np.tile(target_world, (num_remaining_waypoints, 1))
                path_world = np.concatenate((path_world, remaining_waypoints), axis=0)

        return path_world, geodesic_distance
