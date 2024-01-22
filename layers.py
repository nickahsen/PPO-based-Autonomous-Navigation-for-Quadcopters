import torch.nn as nn
import MinkowskiEngine as ME
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym
import torch
# Define a simple Minkowski Network for processing 3D data
class MinkowskiNet(ME.MinkowskiNetwork):
    def __init__(self):
        super(MinkowskiNet, self).__init__(3)  # 3D space
        self.conv1 = ME.MinkowskiConvolution(1, 16, kernel_size=3, stride=2, dimension=3)
        self.conv2 = ME.MinkowskiConvolution(16, 32, kernel_size=3, stride=2, dimension=3)
        self.bn1 = ME.MinkowskiBatchNorm(16)
        self.bn2 = ME.MinkowskiBatchNorm(32)
        self.relu = ME.MinkowskiReLU(inplace=True)
        self.pool = ME.MinkowskiGlobalMaxPooling()

    def forward(self, x):
        x = self.conv1(x)
        if x.F.shape[0] > 1:  # Apply batch norm only if batch size is greater than 1
            x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        if x.F.shape[0] > 1:
            x = self.bn2(x)
        x = self.relu(x)

        x = self.pool(x)
        return x

class CustomPolicy(BaseFeaturesExtractor):

    def __init__(self, observation_space: gym.spaces.Dict, features_dim):
        super(CustomPolicy, self).__init__(observation_space, features_dim)

        self.minkowski_net = MinkowskiNet()

    def forward(self, observations):
        point_cloud = observations['pcl_obs']  # Shape: [64, 1000, 3]
        goal = observations['goal_obs']        # Make sure this is compatible

        processed_points_list = []
        for i in range(point_cloud.shape[0]):  # Iterate over each point cloud in the batch
            single_point_cloud = point_cloud[i]  # Shape: [1000, 3]
            coordinates = ME.utils.batched_coordinates([single_point_cloud])
            features = torch.ones(coordinates.shape[0], 1)
            sparse_tensor = ME.SparseTensor(features, coordinates=coordinates)
            processed_points = self.minkowski_net(sparse_tensor).F
            processed_points_list.append(processed_points)

        # Depending on your next steps, concatenate or stack these processed points
        processed_points_batch = torch.cat(processed_points_list, dim=0)

        # Concatenate with goal here if needed
        # Ensure dimensions are compatible for concatenation
        final_output = torch.cat([processed_points_batch, goal], dim=1)

        return final_output

    # def forward(self, observations):
    #     point_cloud = observations['pcl_obs']
    #     goal = observations['goal_obs']

    #     point_cloud = point_cloud.squeeze(0)

    #     # Process point cloud with MinkowskiNet
    #     print("point cloud shape = ",  point_cloud.shape)
    #     coordinates = ME.utils.batched_coordinates([point_cloud])
    #     features = torch.ones(coordinates.shape[0], 1)
    #     sparse_tensor = ME.SparseTensor(features, coordinates=coordinates)
    #     processed_points = self.minkowski_net(sparse_tensor).F

    #     return torch.cat([processed_points, goal], dim=1)
        