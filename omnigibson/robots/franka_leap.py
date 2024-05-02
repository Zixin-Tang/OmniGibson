import os
from typing import Dict, Iterable

import numpy as np

import omnigibson.utils.transform_utils as T
from omnigibson.macros import gm
from omnigibson.robots.manipulation_robot import GraspingPoint, ManipulationRobot


class FrankaLeap(ManipulationRobot):
    """
    Franka Robot with Leap right hand
    """

    def __init__(
        self,
        # Shared kwargs in hierarchy
        name,
        hand="right",
        prim_path=None,
        uuid=None,
        scale=None,
        visible=True,
        visual_only=False,
        self_collisions=True,
        load_config=None,
        fixed_base=True,
        # Unique to USDObject hierarchy
        abilities=None,
        # Unique to ControllableObject hierarchy
        control_freq=None,
        controller_config=None,
        action_type="continuous",
        action_normalize=True,
        reset_joint_pos=None,
        # Unique to BaseRobot
        obs_modalities="all",
        proprio_obs="default",
        sensor_config=None,
        # Unique to ManipulationRobot
        grasping_mode="physical",
        **kwargs,
    ):
        """
        Args:
            name (str): Name for the object. Names need to be unique per scene
            hand (str): One of {"left", "right"} - which hand to use, default is right
            prim_path (None or str): global path in the stage to this object. If not specified, will automatically be
                created at /World/<name>
            uuid (None or int): Unique unsigned-integer identifier to assign to this object (max 8-numbers).
                If None is specified, then it will be auto-generated
            scale (None or float or 3-array): if specified, sets either the uniform (float) or x,y,z (3-array) scale
                for this object. A single number corresponds to uniform scaling along the x,y,z axes, whereas a
                3-array specifies per-axis scaling.
            visible (bool): whether to render this object or not in the stage
            visual_only (bool): Whether this object should be visual only (and not collide with any other objects)
            self_collisions (bool): Whether to enable self collisions for this object
            load_config (None or dict): If specified, should contain keyword-mapped values that are relevant for
                loading this prim at runtime.
            abilities (None or dict): If specified, manually adds specific object states to this object. It should be
                a dict in the form of {ability: {param: value}} containing object abilities and parameters to pass to
                the object state instance constructor.
            control_freq (float): control frequency (in Hz) at which to control the object. If set to be None,
                simulator.import_object will automatically set the control frequency to be 1 / render_timestep by default.
            controller_config (None or dict): nested dictionary mapping controller name(s) to specific controller
                configurations for this object. This will override any default values specified by this class.
            action_type (str): one of {discrete, continuous} - what type of action space to use
            action_normalize (bool): whether to normalize inputted actions. This will override any default values
                specified by this class.
            reset_joint_pos (None or n-array): if specified, should be the joint positions that the object should
                be set to during a reset. If None (default), self._default_joint_pos will be used instead.
                Note that _default_joint_pos are hardcoded & precomputed, and thus should not be modified by the user.
                Set this value instead if you want to initialize the robot with a different rese joint position.
            obs_modalities (str or list of str): Observation modalities to use for this robot. Default is "all", which
                corresponds to all modalities being used.
                Otherwise, valid options should be part of omnigibson.sensors.ALL_SENSOR_MODALITIES.
                Note: If @sensor_config explicitly specifies `modalities` for a given sensor class, it will
                    override any values specified from @obs_modalities!
            proprio_obs (str or list of str): proprioception observation key(s) to use for generating proprioceptive
                observations. If str, should be exactly "default" -- this results in the default proprioception
                observations being used, as defined by self.default_proprio_obs. See self._get_proprioception_dict
                for valid key choices
            sensor_config (None or dict): nested dictionary mapping sensor class name(s) to specific sensor
                configurations for this object. This will override any default values specified by this class.
            grasping_mode (str): One of {"physical", "assisted", "sticky"}.
                If "physical", no assistive grasping will be applied (relies on contact friction + finger force).
                If "assisted", will magnetize any object touching and within the gripper's fingers.
                If "sticky", will magnetize any object touching the gripper's fingers.
            kwargs (dict): Additional keyword arguments that are used for other super() calls from subclasses, allowing
                for flexible compositions of various object subclasses (e.g.: Robot is USDObject + ControllableObject).
        """

        self.hand = hand
        # Run super init
        super().__init__(
            prim_path=prim_path,
            name=name,
            uuid=uuid,
            scale=scale,
            visible=visible,
            fixed_base=fixed_base,
            visual_only=visual_only,
            self_collisions=self_collisions,
            load_config=load_config,
            abilities=abilities,
            control_freq=control_freq,
            controller_config=controller_config,
            action_type=action_type,
            action_normalize=action_normalize,
            reset_joint_pos=reset_joint_pos,
            obs_modalities=obs_modalities,
            proprio_obs=proprio_obs,
            sensor_config=sensor_config,
            grasping_mode=grasping_mode,
            grasping_direction="upper",
            **kwargs,
        )

    @property
    def model_name(self):
        return f"FrankaLeap{self.hand.capitalize()}"

    @property
    def discrete_action_list(self):
        # Not supported for this robot
        raise NotImplementedError()

    def _create_discrete_action_space(self):
        # Fetch does not support discrete actions
        raise ValueError("Franka does not support discrete actions!")

    @property
    def controller_order(self):
        return ["arm_{}".format(self.default_arm), "gripper_{}".format(self.default_arm)]

    @property
    def _default_controllers(self):
        controllers = super()._default_controllers
        controllers["arm_{}".format(self.default_arm)] = "InverseKinematicsController"
        controllers["gripper_{}".format(self.default_arm)] = "MultiFingerGripperController"
        return controllers

    @property
    def _default_gripper_multi_finger_controller_configs(self):
        conf = super()._default_gripper_multi_finger_controller_configs
        conf[self.default_arm]["mode"] = "independent"
        conf[self.default_arm]["command_input_limits"] = None
        return conf

    @property
    def _default_joint_pos(self):
        # position where the hand is parallel to the ground
        return np.r_[[0.86, -0.27, -0.68, -1.52, -0.18, 1.29, 1.72], np.zeros(16)]

    @property
    def assisted_grasp_start_points(self):
        return {
            self.default_arm: [
                GraspingPoint(link_name=f"palm_center", position=[0, -0.025, 0.035]),
                GraspingPoint(link_name=f"palm_center", position=[0, 0.03, 0.035]),
                GraspingPoint(link_name=f"fingertip_4", position=[-0.0115, -0.07, -0.015]),
            ]
        }

    @property
    def assisted_grasp_end_points(self):
        return {
            self.default_arm: [
                GraspingPoint(link_name=f"fingertip_1", position=[-0.0115, -0.06, 0.015]),
                GraspingPoint(link_name=f"fingertip_2", position=[-0.0115, -0.06, 0.015]),
                GraspingPoint(link_name=f"fingertip_3", position=[-0.0115, -0.06, 0.015]),
            ]
        }

    @property
    def finger_lengths(self):
        return {self.default_arm: 0.1}

    @property
    def arm_control_idx(self):
        return {self.default_arm: np.arange(7)}

    @property
    def gripper_control_idx(self):
        # thumb.proximal, ..., thumb.tip, ..., ring.tip
        return {self.default_arm: np.array([8, 12, 16, 20, 7, 11, 15, 19, 9, 13, 17, 21, 10, 14, 18, 22])}

    @property
    def arm_link_names(self):
        return {self.default_arm: [f"panda_link{i}" for i in range(8)]}

    @property
    def arm_joint_names(self):
        return {self.default_arm: [f"panda_joint_{i+1}" for i in range(7)]}

    @property
    def eef_link_names(self):
        return {self.default_arm: "palm_center"}

    @property
    def finger_link_names(self):
        links = ["mcp_joint", "pip", "dip", "fingertip", "realtip"]
        return {self.default_arm: [f"{link}_{i}" for i in range(1, 5) for link in links]}

    @property
    def finger_joint_names(self):
        # thumb.proximal, ..., thumb.tip, ..., ring.tip
        return {self.default_arm: [f"finger_joint_{i}" for i in [12, 13, 14, 15, 1, 0, 2, 3, 5, 4, 6, 7, 9, 8, 10, 11]]}

    @property
    def usd_path(self):
        return os.path.join(gm.ASSET_PATH, f"models/franka/franka_leap_{self.hand}.usd")

    @property
    def robot_arm_descriptor_yamls(self):
        return {self.default_arm: os.path.join(gm.ASSET_PATH, "models/franka/franka_leap_description.yaml")}

    @property
    def urdf_path(self):
        return os.path.join(gm.ASSET_PATH, f"models/franka/franka_leap_{self.hand}.urdf")

    @property
    def teleop_rotation_offset(self):
        return {self.default_arm: T.euler2quat(np.array([0, np.pi, np.pi / 2]))}

    def _handle_assisted_grasping(self):
        """
        Handles assisted grasping by creating or removing constraints.
        Note that we need to overwrite this because the rotation of finger along its length does not contribute to the grasping behavior.
        """
        # Loop over all arms
        for arm in self.arm_names:
            # We apply a threshold based on the control rather than the command here so that the behavior
            # stays the same across different controllers and control modes (absolute / delta). This way,
            # a zero action will actually keep the AG setting where it already is.
            controller = self._controllers[f"gripper_{arm}"]
            controlled_joints = controller.dof_idx
            threshold = np.mean(
                [self.joint_lower_limits[controlled_joints], self.joint_upper_limits[controlled_joints]], axis=0
            )
            if controller.control is None:
                applying_grasp = False
            elif self._grasping_direction == "lower":
                applying_grasp = np.any(np.delete(controller.control < threshold, [5, 9, 13]))
            else:
                applying_grasp = np.any(np.delete(controller.control > threshold, [5, 9, 13]))
            # Execute gradual release of object
            if self._ag_obj_in_hand[arm]:
                if self._ag_release_counter[arm] is not None:
                    self._handle_release_window(arm=arm)
                else:
                    if gm.AG_CLOTH:
                        self._update_constraint_cloth(arm=arm)

                    if not applying_grasp:
                        self._release_grasp(arm=arm)
            elif applying_grasp:
                self._establish_grasp(arm=arm, ag_data=self._calculate_in_hand_object(arm=arm))
