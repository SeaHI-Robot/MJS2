import yaml
import argparse
from typing import Dict
import numpy as np


def load_config(config_file, robot_ns) -> Dict:
    """Load configuration parameters from a YAML file for a robot type.

    Args:
        cfg (string): Path of config file.
        robot_ns (string): Robot type (namespace).

    Returns:
        Dict: A dictionary that stores all the configuration data
    """
    with open(config_file) as f:
        config = yaml.safe_load(f)

    # Create a dictionary to hold configuration parameters
    config_params = {}

    # Policy files
    config_params["policy_files"] = config[robot_ns]["policy_files"]

    # Default joint angle
    config_params["default_joint_angle"] = config[robot_ns]["init_state"]["default_joint_angle"]

    # Initial base height
    config_params["init_base_height"] = config[robot_ns]["init_state"]["init_base_height"]
    config_params["init_commands"] = config[robot_ns]["init_state"]["init_commands"]

    # Data size
    config_params["num_action"] = config[robot_ns]["size"]["num_action"]
    config_params["num_obs"] = config[robot_ns]["size"]["num_obs"]
    config_params["num_user_cmd"] = config[robot_ns]["size"]["num_user_cmd"]
    config_params["obs_history_ids"] = config[robot_ns]["size"]["obs_history_ids"]

    # Control
    config_params["kp"] = config[robot_ns]["control"]["stiffness"]
    config_params["kd"] = config[robot_ns]["control"]["damping"]
    config_params["torque_limit"] = config[robot_ns]["control"]["torque_limit"]
    config_params["decimation"] = config[robot_ns]["control"]["decimation"]
    config_params["action_scale"] = config[robot_ns]["control"]["action_scale"]
    config_params["gravity"] = config[robot_ns]["control"]["gravity"]

    # Normalization
    config_params["clip_obs"] = config[robot_ns]["normalization"]["clip_scales"]["clip_obs"]
    config_params["clip_action"] = config[robot_ns]["normalization"]["clip_scales"]["clip_action"]
    config_params["obs_scales"] = config[robot_ns]["normalization"]["obs_scales"]
    config_params["hip_scales"] = config[robot_ns]["normalization"]["clip_scales"]["clip_action"]

    # Noise
    config_params["add_noise"] = config[robot_ns]["noise"]["add_noise"]
    config_params["noise_scales"] = config[robot_ns]["noise"]["noise_scales"]

    return config_params


def get_args(policy_files: Dict) -> argparse.Namespace:
    """Get Script Policy Args

    Args:
        policy_files (Dict): A dictionary that stores all the policies

        Returns:
            argparse.Namespace: Parsed arguments with valid policy name
    """
    parser = argparse.ArgumentParser(description="Process Args for Policy File.")
    parser.add_argument(
        "-p",
        "--policy",
        choices=list(policy_files.keys()),
        default=list(policy_files.keys())[0],
        help=f"Policy to be loaded. Available choices: {list(policy_files.keys())}",
        type=str,
    )
    args = parser.parse_args()

    # Validate policy choice
    if args.policy not in policy_files:
        raise ValueError("ERROR: Invalid task name: ", args.policy)

    return args


class ObservationBuffer:
    def __init__(self, num_obs, include_history_steps):
        self.num_obs = num_obs
        self.include_history_steps = include_history_steps
        self.num_obs_total = num_obs * include_history_steps
        self.obs_buf = np.zeros([1, self.num_obs_total])

    #  def reset(self, reset_idxs, new_obs):
    #      """Reset observation buffer"""
    #      self.obs_buf[reset_idxs] = new_obs.repeat(1, self.include_history_steps)
    def reset(self):
        """Reset observation buffer"""
        self.obs_buf = np.zeros([1, self.num_obs_total])

    def insert(self, new_obs):
        """Shift old obs to the front. Insert new obs at the back.

        Example:
           [num_obs<i-3>, num_obs<i-2>, num_obs<i-1>] --> [num_obs<i-2>, num_obs<i-1>, num_obs<i>] 

        Args:
            new_obs: New observation tensor to be added to the buffer.
                    Shape: (1, num_obs)
        """
        # Shift old observations to the front
        #  self.obs_buf[:, :-self.num_obs] = self.obs_buf[:, self.num_obs:].copy()
        self.obs_buf[:, : self.num_obs * (self.include_history_steps - 1)] = self.obs_buf[:,self.num_obs : self.num_obs * self.include_history_steps].copy()

        # Add new observation at the end
        self.obs_buf[:, -self.num_obs :] = new_obs

    def get_obs_vec(self, obs_ids):
        """Gets history of observations indexed by obs_ids.

        Arguments:
            obs_ids: An array of integers with which to index the desired
                observations, where 0 is the latest observation and
                include_history_steps - 1 is the oldest observation.
        """

        obs = []
        for obs_id in reversed(sorted(obs_ids)):
        #  for obs_id in reversed(obs_ids):
            slice_idx = self.include_history_steps - obs_id - 1
            obs.append(self.obs_buf[:, slice_idx * self.num_obs : (slice_idx + 1) * self.num_obs])
        return np.concatenate(obs, axis=-1)
