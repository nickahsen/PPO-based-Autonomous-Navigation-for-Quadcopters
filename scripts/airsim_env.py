from typing import Any, List, Tuple
from dataclasses import dataclass
import gymnasium as gym
import numpy as np
from collections import OrderedDict
import os
from . import airsim
from scripts import binvox_rw
import random
from scipy.spatial.transform import Rotation as R
import pprint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
@dataclass
class Section:
    target: List[float]
    offset: List[float]


@dataclass
class TrainConfig:
    sections: List[Section]

    def __post_init__(self):
        if isinstance(self.sections[0], dict):
            self.sections = [Section(**sec) for sec in self.sections]


class AirSimDroneEnv(gym.Env):
    def __init__(
        self,
        ip_address: str,
        env_config: TrainConfig,
    ):
        self.sections = env_config.sections

        # Run this for the first time to create a voxel grid map. TODO: If map doesn't exist, run this
        #self.create_voxel_grid()

        # Get the current directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        binvox_path = os.path.join(current_dir, '../maps', 'venice_map.binvox')
        
        with open(binvox_path, 'rb') as f:
            self.map = binvox_rw.read_as_3d_array(f)        

        # self.identify_and_fill_hollow_spaces(self.map)

        self.drone = airsim.MultirotorClient(ip=ip_address)
        rgb_shape = self.get_rgb_image().shape

        pcl_shape = (1000, 3)
        # gym.spaces.Box is a class provided by the OpenAI Gym library that
        # represents a continuous space in a reinforcement learning environment.
        # In RL, a "space" defines the possible values that state and action
        # variables can take. The Box space is used when these variables can
        # take on any real-valued number within a specified range.

        # The space is image whose shape is (50, 50, 3) and value range is
        # [0, 255];
        observation_space = OrderedDict()
        observation_space["goal_obs"] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)
        #observation_space["image_obs"] = gym.spaces.Box(low=0, high=255, shape=rgb_shape, dtype=np.uint8)
        observation_space["pcl_obs"] = gym.spaces.Box(low=0, high=255, shape=pcl_shape, dtype=np.float32)
        self.observation_space = gym.spaces.Dict(observation_space)

        # For instance, if you were working with a grid-world environment, these
        # nine discrete actions might correspond to moving in different
        # directions (e.g., up, down, left, right, or diagonally) or taking
        # specific actions within the environment.
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        self.random_start = True
        #self.setup_flight()
        self.steps = 0
        self.target_dist_prev = 0.0
        self.collision_time = -1

        # Camera parameters for PCL
        Width = 256
        Height = 192
        CameraFOV = 90
        self.Fx = self.Fy = Width / (2 * math.tan(CameraFOV * math.pi / 360))
        self.Cx = Width / 2
        self.Cy = Height / 2
    
    def identify_and_fill_hollow_spaces(self, map):
        voxel_data = map.data
        visited = np.zeros_like(voxel_data, dtype=bool)
        nx, ny, nz = voxel_data.shape

        # Iterative flood fill using a stack
        def flood_fill_iterative():
            stack = []

            # Helper function to add neighbors to the stack
            def add_neighbors(x, y, z):
                for dx, dy, dz in [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]:
                    nx, ny, nz = x + dx, y + dy, z + dz
                    if 0 <= nx < voxel_data.shape[0] and 0 <= ny < voxel_data.shape[1] and 0 <= nz < voxel_data.shape[2]:
                        if not visited[nx, ny, nz] and not voxel_data[nx, ny, nz]:
                            stack.append((nx, ny, nz))

            # Add initial edge voxels
            for x in range(nx):
                stack.append((x, 0, 0))
                stack.append((x, ny - 1, 0))
                stack.append((x, 0, nz - 1))
                stack.append((x, ny - 1, nz - 1))
            for y in range(ny):
                stack.append((0, y, 0))
                stack.append((nx - 1, y, 0))
                stack.append((0, y, nz - 1))
                stack.append((nx - 1, y, nz - 1))
            for z in range(nz):
                stack.append((0, 0, z))
                stack.append((nx - 1, 0, z))
                stack.append((0, ny - 1, z))
                stack.append((nx - 1, ny - 1, z))

            # Process the stack
            while stack:
                x, y, z = stack.pop()
                if not visited[x, y, z]:
                    visited[x, y, z] = True
                    add_neighbors(x, y, z)

        flood_fill_iterative()

        # Fill all unvisited, empty voxels
        map.data[~visited & ~voxel_data] = True

        output_path = os.path.join(os.getcwd(), "filled.binvox")
        with open(output_path, 'wb') as f:
            print("map filled")
            binvox_rw.write(map, f)


    def create_voxel_grid(self):

        client = airsim.VehicleClient()
        center = airsim.Vector3r(0, 0, 0)
        voxel_size = 100
        res = 1

        output_path = os.path.join(os.getcwd(), "block.binvox")
        client.simCreateVoxelGrid(center, 200, 200, 100, res, output_path)
        print("voxel map generated!")

        with open(output_path, 'rb') as f:
            map = binvox_rw.read_as_3d_array(f)   
        # Set every below ground level as "occupied". #TODO: add inflation to the map
        map.data[:,:,:50] = True
        map.data[:,:,80:] = True
        binvox_edited_path = os.path.join(os.getcwd(), "block_edited.binvox")
        with open(binvox_edited_path, 'wb') as f:
            binvox_rw.write(map, f)

    def interpolate(self, start, end, t):
        """Linearly interpolate between start and end points."""
        return [start[i] + t * (end[i] - start[i]) for i in range(3)]


    def check_obstacle_between(self, start, end, occupancy_grid, debug=False):
        """
        Check if there is an obstacle between two points in a 3D grid.
        Uses a simplified line traversal algorithm.
        If debug is True, it also plots the current point for visualization.
        """
        # Function to plot the current point for debugging
        def plot_debug_point(start, end, current):
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            # Start and end points
            ax.scatter(start[0], start[1], start[2], color="green", label="Start")
            ax.scatter(end[0], end[1], end[2], color="red", label="End")

            # Current point
            ax.scatter(current[0], current[1], current[2], color="blue", label="Current")

            # Labels and legend
            ax.set_xlabel('X axis')
            ax.set_ylabel('Y axis')
            ax.set_zlabel('Z axis')
            ax.legend()

            plt.show()

        # Calculate total number of steps needed for the interpolation
        num_steps = int(np.linalg.norm(np.array(end) - np.array(start)) * 2)  # x2 for higher resolution

        for step in range(num_steps):
            t = step / float(num_steps)
            current = self.interpolate(start, end, t)
            grid_point = [int(round(coord)) for coord in current]

            # Check bounds
            if any(coord < 0 or coord >= occupancy_grid.shape[i] for i, coord in enumerate(grid_point)):
                continue  # Skip points outside the grid

            

            # Check if current grid point is an obstacle
            if occupancy_grid[tuple(grid_point)] != 0:
                # Plot the current point for debugging
                if debug:
                    plot_debug_point(start, end, current)
                return True  # Obstacle found

        return False  # No obstacle found


    def sample_start_goal_pos(self):

        # find free space points
        free_space_points = np.argwhere(self.map.data == 0)

        min_distance = 15
        max_distance = 25

        iter = 0 
        
        while True:
            iter += 1
            #print("iteration 2 = " , iter)
            start_point = random.choice(free_space_points)
            end_point = random.choice(free_space_points)

            if min_distance <= np.linalg.norm(np.array(start_point) - np.array(end_point)) <= max_distance:
                if self.check_obstacle_between(start_point, end_point, self.map.data):
                    start_pos = [start_point[1] + self.map.translate[0] , start_point[0] + self.map.translate[1], abs(self.map.translate[2]) - start_point[2]]
                    goal_pos = [end_point[1] + self.map.translate[0] , end_point[0] + self.map.translate[1], abs(self.map.translate[2]) - end_point[2]]

                     # # For visualization
                    # asset_name = 'Sphere'
                    
                    # scale = airsim.Vector3r(0.2, 0.2, 0.2)
                    # desired_name = f"{asset_name}_spawn_{random.randint(0, 100)}"
                    # pose = airsim.Pose(position_val=airsim.Vector3r(goal_pos[0], goal_pos[1], goal_pos[2]))

                    # obj_name = self.drone.simSpawnObject(desired_name, asset_name, pose, scale, False)

                    # print(f"Created object {obj_name} from asset {asset_name} "
                    #     f"at pose {pose}, scale {scale}")

                    return start_pos, goal_pos
                    

        # free_space_points = np.argwhere(self.map.data == 0)
        # x = random.choice(free_space_points)

        # indx = np.where(self.map.data == 0)

        # a = random.randint(0 , len(indx[0]))
        # start_x = indx[0][a]
        # start_y = indx[1][a]
        # start_z = indx[2][a]
        # #rint(f"x : {start_x}, y: {start_y}")

        # start_pos = [start_y + self.map.translate[0] , start_x + self.map.translate[1], abs(self.map.translate[2]) - start_z]
        # #start_pos = [1, 0, 0]
        # # Set relative position and orientation wrt to the start, not 100% correct but can't be bothered
        # # x = [-1,1][random.randrange(2)]
        # # y = [-1,1][random.randrange(2)]
        # # z = [-1,1][random.randrange(2)]

        # x = [0,1][random.randrange(2)]
        # y = [0,1][random.randrange(2)]
        # #relative_pos = [x*random.uniform(4, 6),  y*random.uniform(4, 6), z*random.uniform(0, 0)] # TODO: need to sample a collision free pos from the map
        # relative_pos = [x*random.uniform(3, 5),  y*random.uniform(3, 5), 0] # TODO: need to sample a collision free pos from the map
        # goal_pos = [x + y for x, y in zip(start_pos, relative_pos)]
        

        # # For visualization
        # asset_name = 'Sphere'
        
        # scale = airsim.Vector3r(0.1, 0.1, 0.1)
        # desired_name = f"{asset_name}_spawn_{random.randint(0, 100)}"
        # pose = airsim.Pose(position_val=airsim.Vector3r(goal_pos[0], goal_pos[1], goal_pos[2]))

        # obj_name = self.drone.simSpawnObject(desired_name, asset_name, pose, scale, False)

        # # print(f"Created object {obj_name} from asset {asset_name} "
        # #     f"at pose {pose}, scale {scale}")

        # return start_pos, goal_pos

    def step(self, action):
        self.do_action(action)
        obs, info = self.get_obs()
        truncated = self.steps > 200
        reward, done, info = self.compute_reward(info)
        
        return obs, reward, done, truncated, info

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ):
        self.setup_flight()
        obs, _ = self.get_obs()
        return obs, {}

    def render(self):
        return self.get_obs()

    def setup_flight(self):
        self.drone.reset()
        self.drone.enableApiControl(True)
        # Arming a drone means preparing it for flight. When you arm a drone,
        # you enable its propulsion system, allowing it to generate thrust and
        # lift off the ground.
        # Disarming a drone means shutting down its propulsion system and
        # disabling its ability to generate thrust. This is typically done when
        # you want to power down or land the drone safely.
        # True to arm, False to disarm the vehicle
        self.drone.armDisarm(True)

        # Prevent drone from falling after reset
        #self.drone.takeoffAsync().join()
        self.drone.moveToZAsync(-1, 1)

        # Get collision time stamp
        self.collision_time = self.drone.simGetCollisionInfo().time_stamp
        
        start_pos, goal_pos = self.sample_start_goal_pos()

        self.agent_start_pos = np.array(start_pos, dtype=np.float64)
        self.target_pos = np.array(goal_pos, dtype=np.float64)

        print("-----------A new flight!------------")
        print(f"Start point is {self.agent_start_pos}")
        print(f"Target point is {self.target_pos}")
        print("-----------Start flying!------------")
        self.steps = 0
        start_x_pos, start_y_pos, start_z_pos = (
            float(self.agent_start_pos[0]),
            float(self.agent_start_pos[1]),
            float(self.agent_start_pos[2]),
        )

        
        # # Start the agent at random section at a random yz position
        # y_pos, z_pos = ((np.random.rand(1,2)-0.5)*2).squeeze()
        # airsim.Pose: This is a class provided by the AirSim library for
        # defining the pose of an object. A pose typically includes information
        # about its position and orientation.
        # airsim.Vector3r(self.agent_start_pos, y_pos, z_pos): This part creates
        # a Vector3r object, which represents a 3D vector. It's used to specify
        # the position of the object. self.agent_start_pos is likely a variable
        # or value representing the x-coordinate, y_pos is the y-coordinate,
        # and z_pos is the z-coordinate.
        pose = airsim.Pose(airsim.Vector3r(*self.agent_start_pos))
        # Set the pose of the vehicle
        self.drone.simSetVehiclePose(pose=pose, ignore_collision=True)
        
        self.drone.moveToZAsync(start_z_pos, 10).join()

        #self.drone.moveToPositionAsync(start_x_pos, start_y_pos, start_z_pos, 5).join()
        # Get target distance for reward calculation
        # This line of code calculates the Euclidean distance between two
        # 2D points: [y_pos, z_pos] and self.target_pos
        # self.target_dist_prev: This variable is assigned the computed distance
        # value. It seems to be used to store the previous distance between the
        # two points, possibly for tracking changes in distance over time.
        self.target_dist_prev = np.linalg.norm(self.agent_start_pos - self.target_pos)
        print(f"target_dist_prev: {self.target_dist_prev}")

    def do_action(self, action):

        # Execute action
        #print(action)
        self.drone.moveByVelocityBodyFrameAsync(float(action[0]), float(action[1]), float(action[2]), duration=0.1).join()

        #self.drone.landAsync()

    def move_to_pos(self, goal):
        self.drone.moveToPositionAsync(*goal, velocity=1.5).join()

    def get_obs(self):
        info = {"collision": self.is_collision()}

        obs = OrderedDict()
        obs["pcl_obs"] = self.get_plc_data()
        goal_obs = self.global_to_local(self.target_pos)
        obs["goal_obs"] = goal_obs

        return obs, info

    @property
    def current_vel(self):
        vel = self.drone.getMultirotorState().kinematics_estimated.linear_velocity
        return np.array([vel.x_val, vel.y_val, vel.z_val])

    def compute_reward(self, info) -> Tuple[float, bool]:
        reward = 0.0
        done = False
        self.steps += 1
        info["is_success"] = False
        info["is_collision"] = False
        info["timeout"] = False


        drone_pose = self.drone.simGetVehiclePose()
        drone_pos = np.array([drone_pose.position.x_val, drone_pose.position.y_val, drone_pose.position.z_val])

        # Target distance based reward
        potential_reward_weight = 0.20 # TODO: add in config file
        target_dist_curr = float(np.linalg.norm(self.target_pos - drone_pos))
        #print("target_dist_curr: ", target_dist_curr)
        reward += (self.target_dist_prev - target_dist_curr) * potential_reward_weight

        self.target_dist_prev = target_dist_curr

        # Goal reward
        goal_threshold = 0.30
        if target_dist_curr < goal_threshold:
            reward += 1
            done = True
            info["is_success"] = True

        # Timestep reward
        reward += -0.005

        # Collision penalty
        if self.is_collision():
            print("The drone has collided with the obstacle!!!")
            reward += -1
            info["is_collision"] = True
            done = True
        elif self.is_landing():
            # Check if the drone's altitude is less than the landing threshold
            print("Drone has touched the ground!!!")
            reward += -1
            done = True
        elif target_dist_curr >= 50:
            print("The drone has flown out of the specified range!!!")
            reward += -1
            done = True
        elif self.steps > 100:
            info["is_timeout"] = True
            reward += -1
            done = True

        if done == True or self.steps % 10 == 0:
            print(f"Steps {self.steps} -> reward: {reward}, done: {done}")
        
        return reward, done, info

    def is_landing(self):
        # Set a threshold for how close the drone should be to the ground
        # to consider it landed
        landing_threshold = -0.1  # You may need to adjust this value
        state = self.drone.getMultirotorState()
        position = state.kinematics_estimated.position
        return position.z_val > landing_threshold

    def is_collision(self):
        current_collision_time = self.drone.simGetCollisionInfo().time_stamp
        if current_collision_time != self.collision_time:
            flag = True
            self.collision_time = self.drone.simGetCollisionInfo().time_stamp
        else:
            flag = False

        return flag

    def is_collided(self):
        flag = self.drone.simGetCollisionInfo().has_collided
        if flag:
            print("collided!!!!")
        return flag

    def get_rgb_image(self) -> np.ndarray:
        rgb_image_request = airsim.ImageRequest(0, airsim.ImageType.Scene, False, False)
        # # camera control
        # # simGetImage returns compressed png in array of bytes
        # # image_type uses one of the ImageType members
        responses = self.drone.simGetImages([rgb_image_request])
        img1d = np.fromstring(  # type: ignore
            responses[0].image_data_uint8, dtype=np.uint8
        )
        try:
            img2d = np.reshape(img1d, (responses[0].height, responses[0].width, 3))
        except:
            np.zeros((144,256,3))

        img_rgb = np.flipud(img2d)
        return img_rgb

    def get_depth_image(self, thresh=2.0) -> np.ndarray:
        depth_image_request = airsim.ImageRequest(
            1, airsim.ImageType.DepthPerspective, True, False
        )
        responses = self.drone.simGetImages([depth_image_request])
        depth_image = np.array(responses[0].image_data_float, dtype=np.float64)
        depth_image = np.reshape(depth_image, (responses[0].height, responses[0].width))
        depth_image[depth_image > thresh] = thresh
        return depth_image

    def get_lidar_data(self):
        lidarData = self.drone.getLidarData()

        while len(lidarData.point_cloud) < 3:
            lidarData = self.drone.getLidarData()
       
        
        points = self.parse_lidarData(lidarData)

        return points 
        #return points
    
    def parse_lidarData(self, data):
        # reshape array of floats to array of [X,Y,Z]
        points = np.array(data.point_cloud, dtype=np.dtype('f4'))
        points = np.reshape(points, (int(points.shape[0]/3), 3))
       
        return points

    
    def point_cloud_to_voxel_grid(self, point_cloud, grid_dims=(1000, 1000, 1000), voxel_resolution=0.1):
        """
        Convert a point cloud to a voxel grid with a fixed resolution and size, accommodating negative values.

        Parameters:
        point_cloud (numpy.ndarray): Nx3 array of point cloud.
        grid_dims (tuple): The dimensions of the voxel grid as (x_dim, y_dim, z_dim).
        voxel_resolution (float): The size of each voxel.

        Returns:
        numpy.ndarray: 3D array representing the voxel grid.
        """

        # Initialize voxel grid
        voxel_grid = np.zeros(grid_dims, dtype=bool)

        # Convert grid_dims to a NumPy array for mathematical operations
        grid_dims_arr = np.array(grid_dims)

        # Define the center offset for the grid
        grid_center_offset = grid_dims_arr * voxel_resolution / 2

        # Process each point in the point cloud
        for point in point_cloud:
            # Adjust point position relative to the grid center
            adjusted_point = point + grid_center_offset

            # Check if the adjusted point is within the bounds of the voxel grid
            if np.all(adjusted_point < grid_dims_arr * voxel_resolution) and np.all(adjusted_point >= 0):
                # Calculate the voxel coordinates
                voxel_coords = (adjusted_point / voxel_resolution).astype(int)

                # Clip voxel_coords to be within the grid dimensions
                voxel_coords = np.clip(voxel_coords, 0, grid_dims_arr - 1)

                voxel_grid[voxel_coords[0], voxel_coords[1], voxel_coords[2]] = True

        return voxel_grid

    def plot_point_cloud(self, point_cloud):
        """
        Plot a 3D point cloud using matplotlib.

        Parameters:
        point_cloud (numpy.ndarray): Nx3 array of point cloud coordinates.
        """

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')

        # Extract the coordinates
        x, y, z = point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2]

        # Plot the points
        ax.scatter(x, y, z, c='blue', marker='o')

        plt.show()

    def visualize_voxel_grid(self, voxel_grid, voxel_resolution=0.5):
        """
        Visualize a 3D voxel grid using matplotlib.

        Parameters:
        voxel_grid (numpy.ndarray): 3D array representing the voxel grid.
        """

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')

        # Extract the coordinates of the occupied voxels
        x, y, z = voxel_grid.nonzero()

        # Scale the coordinates to original units
        x = x * voxel_resolution
        y = y * voxel_resolution
        z = z * voxel_resolution

        # Plot each voxel as a point
        ax.scatter(x, y, z, zdir='z', c='red', marker='s')

        plt.show()

    def global_to_local(self, pos):
        """
        Convert a 3D point in global frame to agent's local frame

        :param env: environment instance
        :param pos: a 3D point in global frame
        :return: the same 3D point in agent's local frame
        """
        drone_pose = self.drone.simGetVehiclePose()
        drone_pos = np.array([drone_pose.position.x_val, drone_pose.position.y_val, drone_pose.position.z_val])
        v = np.array(pos) - drone_pos

        local_to_global = R.from_quat([drone_pose.orientation.x_val, drone_pose.orientation.y_val, 
            drone_pose.orientation.z_val, drone_pose.orientation.w_val]).as_matrix()
        global_to_local = local_to_global.T
        return np.dot(global_to_local, v)
    
    def get_plc_data(self):
        
        responses = self.drone.simGetImages(
            [airsim.ImageRequest('0', airsim.ImageType.DepthPerspective, pixels_as_float=True, compress=False)])
        response = responses[0]
        img1d = np.array(response.image_data_float, dtype=float)
        img1d[img1d > 255] = 255
        img2d = np.reshape(img1d, (responses[0].height, responses[0].width))
        img2d_converted = self.depthConversion(img2d, self.Fx)
        pcl = self.generatepointcloud(img2d_converted)
        pcl = pcl.reshape(pcl.shape[0]*pcl.shape[1], pcl.shape[2])
        
        num_points = 1000
        # Calculate Euclidean distances
        distances = np.linalg.norm(pcl, axis=1)
        
        # Get the indices of the sorted distances
        sorted_indices = np.argsort(distances)

        # Select the indices of the closest num_points points
        closest_indices = sorted_indices[:num_points]

        # Return 1000 closest points
        p = pcl[closest_indices]
        if p.shape == (1000, 3) and p.size != 0:
            #print("The array is of shape (1000, 3) and is not empty.")
            return p
        else:
            print("Error: Failed to obtain point cloud")
            return np.zeros((1000, 3))

    
    def depthConversion(self, PointDepth, f):
        H = PointDepth.shape[0]
        W = PointDepth.shape[1]
        i_c = float(H) / 2 - 1
        j_c = float(W) / 2 - 1
        columns, rows = np.meshgrid(np.linspace(0, W-1, num=W), np.linspace(0, H-1, num=H))
        DistanceFromCenter = ((rows - i_c)**2 + (columns - j_c)**2)**(0.5)
        PlaneDepth = PointDepth / (1 + (DistanceFromCenter / f)**2)**(0.5)
        return PlaneDepth

    def generatepointcloud(self, depth):
        rows, cols = depth.shape
        c, r = np.meshgrid(np.arange(cols), np.arange(rows), sparse=True)
        valid = (depth > 0) & (depth < 255)
        z = 1000 * np.where(valid, depth / 256.0, np.nan)
        x = np.where(valid, z * (c - self.Cx) / self.Fx, 0)
        y = np.where(valid, z * (r - self.Cy) / self.Fy, 0)
        return np.dstack((x, y, z))


class TestEnv(AirSimDroneEnv):
    def __init__(self, ip_address, env_config):
        self.eps_n = 0
        super().__init__(ip_address, env_config)
        self.agent_traveled = []
        self.random_start = False

    def setup_flight(self):
        super().setup_flight()
        self.eps_n += 1

        # Start the agent at a random yz position
        # y_pos, z_pos = (0, 0)
        pose = airsim.Pose(airsim.Vector3r(*self.agent_start_pos))
        self.drone.simSetVehiclePose(pose=pose, ignore_collision=True)

    def compute_reward(self):
        reward = 0
        done = 0

        x, _, _ = self.current_pose

        if self.is_collision():
            done = 1
            self.agent_traveled.append(x)

        if done and self.eps_n % 5 == 0:
            print("---------------------------------")
            print("> Total episodes:", self.eps_n)
            print(f"> Flight distance (mean): {np.mean(self.agent_traveled):.2f}")
            print("> Holes reached (max):", int(np.max(self.agent_traveled) // 4))
            print("> Holes reached (mean):", int(np.mean(self.agent_traveled) // 4))
            print("---------------------------------\n")

        return reward, done
