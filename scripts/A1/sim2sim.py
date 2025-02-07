import os
import time

import mujoco
import mujoco.viewer
import numpy as np
import torch
import utils
from pynput import keyboard
from scipy.spatial.transform import Rotation as R

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class MJS2:
    """Main class of MuJoco sim2sim: Setting up Configurations

    Args:
        robot_ns (string): Robot namespace.
        robot_model (string): Robot model name (mujoco).

    Usage Example:
        sim = MJS2(robot_ns="MyRobot", robot_model="MyRobot")
        sim.run()
    """

    def __init__(self, robot_ns, robot_model):
        """Class constructor to set up simulation configuration and control policy."""
        # Load Configs
        config_file = os.path.join(BASE_DIR, "cfg", "{}.yaml".format(robot_ns))
        self.cfg = utils.load_config(config_file, robot_ns)

        # Get Policy Args & Load Policy
        self.args = utils.get_args(self.cfg["policy_files"])
        policy_file = os.path.join(BASE_DIR, "policy", robot_ns, self.cfg["policy_files"][self.args.policy])
        self.policy = torch.jit.load(policy_file)

        # Print Startup Message
        print(f'\n\033[32m ===== Loading Policy "{self.args.policy}" from {policy_file} ===== \033[0m\n')
        if self.cfg["add_noise"]:
            print(f"\033[33m ===== Noise is Added ===== \033[0m\n")
        else:
            print(f"\033[33m ===== Not Adding Noise ===== \033[0m\n")

        # Load the MuJoCo model
        self.model = mujoco.MjModel.from_xml_path(os.path.join(BASE_DIR, "resources", robot_model, "scene.xml"))
        self.data = mujoco.MjData(self.model)

        # Control and Simulation parameters
        self.init_base_height = self.cfg["init_base_height"]
        self.num_joint = self.cfg["num_action"]
        self.Kp = self.cfg["kp"] * np.ones(self.num_joint)
        self.Kd = self.cfg["kd"] * np.ones(self.num_joint)
        self.joint_pos_default = np.array(list(self.cfg["default_joint_angle"].values()))
        self.torque_limit = self.cfg["torque_limit"] * np.ones(self.num_joint)
        self.obs_scales = self.cfg["obs_scales"]
        self.obs_cmd_scale = np.array(
            [
                self.obs_scales["lin_vel"],
                self.obs_scales["lin_vel"],
                self.obs_scales["ang_vel"],
            ]
        )
        self.decimation = self.cfg["decimation"]
        self.action_scale = self.cfg["action_scale"]

        # Keycallback function for mujoco viewer
        self.PAUSE = False  # pause simulation
        self.QUIT = False  # quit simulation

        def viewer_key_callback(keycode):
            if chr(keycode) == " ":
                self.PAUSE = not self.PAUSE

        # Reset Simulation and Start the simulation viewer
        self.reset_sim()
        self.viewer = mujoco.viewer.launch_passive(
            self.model,
            self.data,
            key_callback=viewer_key_callback,
            show_left_ui=False,
            show_right_ui=False,
        )

    def reset_sim(self):
        """Reset Simulation Configuration"""
        # Reset mujoco data structs
        mujoco.mj_resetData(self.model, self.data)
        # Set gravity
        self.model.opt.gravity[:] = np.array([0.0, 0.0, self.cfg["gravity"]])
        # Reset robot qpos
        self.data.qpos[0] = 0.0
        self.data.qpos[1] = 0.0
        self.data.qpos[2] = self.init_base_height
        self.data.qpos[3] = 1.0
        self.data.qpos[4] = 0.0
        self.data.qpos[5] = 0.0
        self.data.qpos[6] = 0.0
        # Debug: Print debug info
        if self.args.debug:
            print(" ===== Debug Info Start ===== \n")
            for i in range(self.model.njnt):
                joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
                print(f"Joint {i}: {joint_name}")
            for i in range(self.model.nu):
                actuator_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
                print(f"Actuator {i}: {actuator_name}")
            for i in range(self.model.nsensor):
                sensor_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_SENSOR, i)
                print(f"Sensor {i}: {sensor_name}")
            print("\n ===== Debug Info End ===== \n")
        for i in range(self.num_joint):
            self.data.qpos[i + 7] = self.joint_pos_default[i]
        mujoco.mj_step(self.model, self.data)
        # Reset User commands
        self.commands = self.cfg["init_commands"]  # x velocity, y velocity, yaw angular velocity
        # Reset Policy
        self.action = np.zeros(self.num_joint)  # Policy action: reference joint position
        self.obs = np.zeros(self.cfg["num_obs"])
        self.iter_ = 0

    def get_joint_state(self):
        """Retrieve the joint position and velocity states."""
        q_ = self.data.qpos
        dq_ = self.data.qvel
        return q_[7:], dq_[6:]

    def get_base_state(self):
        """Get the base state including, angular velocity, and projected gravity."""
        quat = self.data.sensor("imu_quat").data[[1, 2, 3, 0]]
        rpy = R.from_quat(quat)
        base_lin_vel = self.data.qvel[0:3]
        base_ang_vel = self.data.sensor("gyro").data
        projected_gravity = rpy.apply(np.array([0.0, 0.0, -1.0]), inverse=True)
        # Update actual base velocity for logging
        self.lin_vel_act = rpy.apply(self.data.qvel[0:3], inverse=True)
        self.ang_vel_act = rpy.apply(self.data.qvel[3:6], inverse=True)
        return base_lin_vel, base_ang_vel, projected_gravity

    def compute_obs(self):
        """Calculate and return the observed states from the policy input."""
        dof_pos, dof_vel = self.get_joint_state()
        base_lin_vel, base_ang_vel, projected_gravity = self.get_base_state()
        # construct obs
        self.obs = np.concatenate(
            [
                base_lin_vel * self.obs_scales["lin_vel"],  # Scaled base linear velocity
                base_ang_vel * self.obs_scales["ang_vel"],  # Scaled base angular velocity
                projected_gravity,  # Projected gravity vector in body frame
                self.obs_cmd_scale * self.commands,  # Scaled velocity commands from user input
                (dof_pos - self.joint_pos_default) * self.obs_scales["dof_pos"],  # Scaled joint positions
                dof_vel * self.obs_scales["dof_vel"],  # Scaled joint velocities
                self.action,  # Last actions taken by the robot
            ]
        )
        # clip observation
        self.obs = np.clip(self.obs, -self.cfg["clip_obs"], self.cfg["clip_obs"])
        # if add noise to obs
        if self.cfg["add_noise"]:
            noise_vec = self.get_noise_scale_vec()
            self.obs += (2 * np.random.rand(*self.obs.shape) - 1) * noise_vec

    def get_noise_scale_vec(self):
        """Gets a vector used to scale the noise added to the observations."""
        noise_vec_ = np.zeros_like(self.obs)
        noise_scales = self.cfg["noise_scales"]
        noise_level = self.cfg["noise_scales"]["noise_level"]
        noise_vec_[:3] = noise_scales["lin_vel"] * noise_level * self.obs_scales["lin_vel"]
        noise_vec_[3:6] = noise_scales["ang_vel"] * noise_level * self.obs_scales["ang_vel"]
        noise_vec_[6:9] = noise_scales["gravity"] * noise_level
        noise_vec_[9:12] = 0.0  # commands
        noise_vec_[12:24] = noise_scales["dof_pos"] * noise_level * self.obs_scales["dof_pos"]
        noise_vec_[24:36] = noise_scales["dof_vel"] * noise_level * self.obs_scales["dof_vel"]
        noise_vec_[36:48] = 0.0  # previous actions
        return noise_vec_

    def compute_actions(self):
        """Computes the actions based on the current observations using the policy session."""
        self.compute_obs()
        input_tensor = np.concatenate([self.obs], axis=0)
        input_tensor = input_tensor.astype(np.float32)
        self.action = self.policy(torch.tensor(input_tensor)).detach().numpy().reshape(-1)
        self.action = np.clip(self.action, -self.cfg["clip_action"], self.cfg["clip_action"])

    def pd_control(self, action_):
        """Use PD to find target joint torques"""
        action_scaled_ = action_ * self.action_scale
        dof_pos, dof_vel = self.get_joint_state()
        return (action_scaled_ + self.joint_pos_default - dof_pos) * self.Kp - self.Kd * dof_vel

    def keyboard_listener_on_press(self, key):
        """User keyboard Interface, in the terminal"""
        try:
            if key.char == "w":
                self.commands[0] += 0.1
            elif key.char == "s":
                self.commands[0] -= 0.1
            elif key.char == "a":
                self.commands[1] += 0.1
            elif key.char == "d":
                self.commands[1] -= 0.1
            elif key.char == "j":
                self.commands[2] += 0.1
            elif key.char == "k":
                self.commands[2] -= 0.1
            elif key.char == "0":
                self.commands[0] = 0.0  # x velocity
                self.commands[1] = 0.0  # y velocity
                self.commands[2] = 0.0  # yaw angular velocity
            elif key.char == "r":
                self.reset_sim()
            elif key.char == "q":
                self.QUIT = True
                raise KeyboardInterrupt
        except AttributeError:
            #  print(f'Press Key: {key}')
            pass

    def run(self):
        """Main loop of simulation"""
        listener = keyboard.Listener(on_press=self.keyboard_listener_on_press)
        listener.start()
        try:
            while self.data.time < 100.0 and self.viewer.is_running():
                if self.QUIT:
                    raise KeyboardInterrupt
                # Main Loop
                if not self.PAUSE:
                    step_start = time.time()
                    if self.iter_ % self.decimation == 0:
                        self.compute_actions()
                    # Generate PD control
                    joint_torque = self.pd_control(self.action)  # Calc torques
                    joint_torque = np.clip(joint_torque, -self.torque_limit, self.torque_limit)  # Clip torques
                    self.data.ctrl = joint_torque
                    if not self.PAUSE:
                        self.viewer.cam.lookat = self.data.body("trunk").subtree_com
                        mujoco.mj_step(self.model, self.data)
                        if self.iter_ % 10 == 0:
                            self.viewer.sync()
                        if self.iter_ % 500 == 0:
                            print(
                                "\033[34mCMD_vel:\033[0m ",
                                self.commands[0],
                                "    ",
                                self.commands[1],
                                "    ",
                                self.commands[2],
                                "    base_height: 0.65",
                            )
                            print(
                                "\033[34mACT_vel:\033[0m ",
                                f"{self.lin_vel_act[0]:.3f}",
                                " ",
                                f"{self.lin_vel_act[1]:.3f}",
                                " ",
                                f"{self.ang_vel_act[2]:.3f}",
                                "    base_height:",
                                self.data.qpos[2],
                            )
                        self.iter_ += 1
                        time_until_next_step = self.model.opt.timestep - (time.time() - step_start)
                        if time_until_next_step > 0:
                            time.sleep(time_until_next_step)
                else:
                    pass
        except KeyboardInterrupt:
            print("\n\033[31m ===== MuJoCo Sim2Sim Program is Stoped ===== \033[0m\n")
            self.viewer.close()
        finally:
            listener.stop()


def main():
    sim = MJS2(robot_ns="A1", robot_model="A1")
    sim.run()


if __name__ == "__main__":
    main()
