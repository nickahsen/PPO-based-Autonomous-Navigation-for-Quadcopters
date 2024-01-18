from typing import Any, Tuple, Dict, Literal
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from collections import OrderedDict

import gymnasium as gym
import numpy as np
import binvox
from scipy.spatial.transform import Rotation as R

from . import airsim
from .util import create_voxel_grid

RGBKEY = Literal["image_obs"]
SEGMENTATIONKEY = Literal["segmentation_obs"]
DEPTHKEY = Literal["depth_obs"]
GOALKEY = Literal["goal_obs"]


@dataclass
class TrainConfig:
    binvox: Path
    use_rgb: bool
    use_segmentation: bool
    use_depth: bool

    def __post_init__(self):
        self.binvox = Path(self.binvox)


class ActionType(Enum):
    DISCRETE = 0
    CONTINUOUS_VELOCITY = 1


class AirSimDroneEnv(gym.Env):
    def __init__(
        self,
        ip_address: str,
        env_config: TrainConfig,
        action_type: ActionType,
        sim_dt: float,
    ):
        self.drone = airsim.MultirotorClient(ip=ip_address)
        rgb_shape = self.get_rgb_image().shape

        self.use_rgb = env_config.use_rgb
        self.use_segmentation = env_config.use_segmentation
        self.use_depth = env_config.use_depth

        self.drone.simPause(True)

        voxel_path = env_config.binvox

        if not voxel_path.exists():
            create_voxel_grid(voxel_path)

        self.voxel = binvox.Binvox.read(str(voxel_path.absolute()), "dense")
        self.voxel.data[:, :, :26] = True

        observation_space = OrderedDict()
        observation_space[GOALKEY] = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32
        )
        if self.use_rgb:
            observation_space[RGBKEY] = gym.spaces.Box(
                low=0, high=255, shape=rgb_shape, dtype=np.uint8
            )
        if self.use_segmentation:
            # TODO: check max
            observation_space[SEGMENTATIONKEY] = gym.spaces.Box(
                low=0, high=255, shape=rgb_shape, dtype=np.uint8
            )
        if self.use_depth:
            # TODO: check max
            observation_space[DEPTHKEY] = gym.spaces.Box(
                low=0, high=255, shape=rgb_shape, dtype=np.uint8
            )

        self.observation_space = gym.spaces.Dict(observation_space)

        self.action_type = action_type
        self.sim_dt = sim_dt
        if self.action_type == ActionType.DISCRETE:
            self.action_space = gym.spaces.Discrete(7)
        elif self.action_type == ActionType.CONTINUOUS_VELOCITY:
            self.action_space = gym.spaces.Box(
                np.array([-1.0, -1.0, -1.0], dtype=np.float32),
                np.array([1.0, 1.0, 1.0], dtype=np.float32),
            )
        else:
            raise ValueError("Invalid Action Type!")

        self.random_start = True
        self.steps = 0
        self.target_dist_prev = 0.0
        self.collision_time = -1
        self.setup_flight()

    def sample_start_goal_pos(self):
        # pylint: disable=invalid-unary-operand-type
        indx = np.where(~self.voxel.numpy())

        a = np.random.randint(len(indx[0]))
        start_x, start_y, start_z = (arr[a] for arr in indx)

        print(f"x : {start_x}, y: {start_y}")

        start_pos = np.array(
            [
                start_x + self.voxel.translate[0],
                start_y + self.voxel.translate[1],
                abs(self.voxel.translate[2]) - start_z,
            ]
        )
        # Drone must start above ground (negative z)
        assert start_pos[2] < 0

        while True:
            a = np.random.randint(len(indx[0]))
            start_x, start_y, start_z = (arr[a] for arr in indx)

            goal_pos = np.array(
                [
                    start_x + self.voxel.translate[0],
                    start_y + self.voxel.translate[1],
                    abs(self.voxel.translate[2]) - start_z,
                ]
            )

            if np.linalg.norm(goal_pos - start_pos) < 10:
                break
        return start_pos, goal_pos

    def step(self, action):
        self.drone.simPause(False)

        if self.action_type == ActionType.DISCRETE:
            pass
        elif self.action_type == ActionType.CONTINUOUS_VELOCITY:
            self.set_velocity(action)
        else:
            raise ValueError("Invalid Action Type!")

        self.drone.simPause(True)

        obs, info = self.get_obs()
        truncated = self.steps > 200
        reward, done, info = self.compute_reward(info)

        return obs, reward, done, truncated, info

    def set_velocity(self, action):
        action = np.clip(action, -1.0, 1.0).tolist()
        _action = [action[0], 0, action[2]]
        a1 = self.drone.rotateByYawRateAsync(
            action[1],
            duration=self.sim_dt,
        )
        a2 = self.drone.moveByVelocityAsync(
            *_action,
            duration=self.sim_dt,
        )
        a1.join()
        a2.join()

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

        # Arming a drone means preparing it for flight.
        self.drone.armDisarm(True)

        # Prevent drone from falling after reset
        # self.drone.takeoffAsync().join()
        self.drone.moveToZAsync(-1, 1)

        # Get collision time stamp
        self.collision_time = self.drone.simGetCollisionInfo().time_stamp

        self.agent_start_pos, self.target_pos = self.sample_start_goal_pos()

        print("-----------A new flight!------------")
        print(f"Start point is {self.agent_start_pos}")
        print(f"Target point is {self.target_pos}")
        print("-----------Start flying!------------")
        self.steps = 0

        pose = airsim.Pose(airsim.Vector3r(*self.agent_start_pos))
        self.drone.simSetVehiclePose(pose=pose, ignore_collision=True)
        self.drone.moveToZAsync(float(self.agent_start_pos[2]), 10).join()
        self.drone.simPause(False)
        self.drone.simPause(True)

        print(f"target_dist_prev: {self.target_dist_prev}")

    def get_obs(self):
        info = {"collision": self.is_collision()}

        obs = OrderedDict()
        if self.use_rgb:
            obs[RGBKEY] = self.get_rgb_image()
        if self.use_segmentation:
            obs[SEGMENTATIONKEY] = self.get_segment_image()
        if self.use_depth:
            obs[DEPTHKEY] = self.get_depth_image()

        goal_obs = self.global_to_local(self.target_pos)
        obs[GOALKEY] = goal_obs

        return obs, info

    @property
    def current_vel(self):
        vel = self.drone.getMultirotorState().kinematics_estimated.linear_velocity
        return np.array([vel.x_val, vel.y_val, vel.z_val])

    def compute_reward(self, info) -> Tuple[float, bool, Dict[str, Any]]:
        reward = 0.0
        done = False
        self.steps += 1
        info["is_success"] = False
        info["is_collision"] = False
        info["timeout"] = False

        drone_pose = self.drone.simGetVehiclePose()
        drone_pos = np.array(
            [
                drone_pose.position.x_val,
                drone_pose.position.y_val,
                drone_pose.position.z_val,
            ]
        )

        # Target distance based reward
        potential_reward_weight = 0.20  # TODO: add in config file
        target_dist_curr = float(np.linalg.norm(self.target_pos - drone_pos))
        # print("target_dist_curr: ", target_dist_curr)
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

        if done or self.steps % 10 == 0:
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
        responses = self.drone.simGetImages([rgb_image_request])
        img1d = np.fromstring(  # type: ignore
            responses[0].image_data_uint8, dtype=np.uint8
        )
        img2d = np.reshape(img1d, (responses[0].height, responses[0].width, 3))

        img_rgb = np.flipud(img2d)
        return img_rgb.astype(np.uint8)

    def get_depth_image(self, thresh=2.0) -> np.ndarray:
        depth_image_request = airsim.ImageRequest(
            0, airsim.ImageType.DepthPerspective, True, False
        )
        responses = self.drone.simGetImages([depth_image_request])
        depth_image = np.array(responses[0].image_data_float, dtype=np.float64)
        depth_image = np.reshape(depth_image, (responses[0].height, responses[0].width))
        depth_image[depth_image > thresh] = thresh
        return depth_image

    def get_segment_image(self) -> np.ndarray:
        segment_request = airsim.ImageRequest(
            0, airsim.ImageType.Segmentation, True, False
        )
        response = self.drone.simGetImages([segment_request])
        img1d = np.fromstring(  # type: ignore
            response[0].image_data_uint8, dtype=np.uint8
        )
        img_seg = img1d.reshape(response.height, response.width, 3)
        img_seg = np.flipud(img_seg)
        return img_seg

    def global_to_local(self, pos):
        """
        Convert a 3D point in global frame to agent's local frame

        :param env: environment instance
        :param pos: a 3D point in global frame
        :return: the same 3D point in agent's local frame
        """
        drone_pose = self.drone.simGetVehiclePose()
        drone_pos = np.array(
            [
                drone_pose.position.x_val,
                drone_pose.position.y_val,
                drone_pose.position.z_val,
            ]
        )
        v = np.array(pos) - drone_pos

        local_to_global = R.from_quat(
            [
                drone_pose.orientation.x_val,
                drone_pose.orientation.y_val,
                drone_pose.orientation.z_val,
                drone_pose.orientation.w_val,
            ]
        ).as_matrix()
        global_to_local = local_to_global.T
        return np.dot(global_to_local, v).astype(np.float32)
