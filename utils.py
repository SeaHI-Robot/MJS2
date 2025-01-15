import yaml
import argparse
from typing import Dict


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

    # Control
    config_params["kp"] = config[robot_ns]["control"]["stiffness"]
    config_params["kd"] = config[robot_ns]["control"]["damping"]
    config_params["torque_limit"] = config[robot_ns]["control"]["torque_limit"]
    config_params["decimation"] = config[robot_ns]["control"]["decimation"]
    config_params["action_scale"] = config[robot_ns]["control"]["action_scale"]
    config_params["gravity"] = config[robot_ns]["control"]["gravity"]

    # Normalization
    config_params["obs_scales"] = config[robot_ns]["normalization"]["obs_scales"]
    config_params["clip_action"] = config[robot_ns]["normalization"]["clip_scales"]["clip_action"]

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
