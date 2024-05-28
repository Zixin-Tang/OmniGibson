import os
import numpy as np

from omnigibson.macros import gm
from omnigibson.controllers import ControlType
from omnigibson.robots.manipulation_robot import ManipulationRobot, GraspingPoint
from omnigibson.robots.locomotion_robot import LocomotionRobot
from omnigibson.robots.active_camera_robot import ActiveCameraRobot
from omnigibson.utils.transform_utils import euler2quat

from omnigibson.macros import gm, create_module_macros

m = create_module_macros(module_path=__file__)

# joint parameters
m.BASE_JOINT_STIFFNESS = 1e6
m.BASE_JOINT_DAMPING = 1e5
m.BASE_JOINT_MAX_EFFORT = 100

m.TORSO_JOINT_STIFFNESS = 100
m.TORSO_JOINT_DAMPING = 1e5
m.TORSO_JOINT_MAX_EFFORT = 1e7
# m.TORSO_JOINT_MAX_VELOCITY = 4*np.pi

m.ARM_JOINT_STIFFNESS = 1e6
m.ARM_JOINT_DAMPING = 1e5
m.ARM_JOINT_MAX_EFFORT = 1e7
# m.FINGER_JOINT_STIFFNESS = 1e3
# m.FINGER_JOINT_MAX_EFFORT = 1e8

m.MAX_LINEAR_VELOCITY = 2  # linear velocity in meters/second
m.MAX_ANGULAR_VELOCITY = 2*np.pi  # angular velocity in radians/second

DEFAULT_ARM_POSES = {
    "straight",
    "tucked"
}

class CURIVersatility(ManipulationRobot, LocomotionRobot, ActiveCameraRobot):
    """
    The CURI robot
    """

    def __init__(
        self,
        # Shared kwargs in hierarchy
        name,
        prim_path=None,
        uuid=None,
        scale=None,
        visible=True,
        visual_only=False,
        self_collisions=True,
        load_config=None,
        fixed_base=False,

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

        # Unique to CURI
        torso_dof=1,
        enable_torso=False,
        default_arm_pose="straight",
        
        **kwargs,
    ):
        """
        Args:
            name (str): Name for the object. Names need to be unique per scene
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
                simulator.import_object will automatically set the control frequency to be at the render frequency by default.
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
            torso_dof (int): Dof of CURI torso. Should be one of:
                {1, 3}
            enable_torso (bool): if False, will prevent the torso from moving during execution.
            default_arm_pose (str): Default pose for the robot arm. Should be one of:
                {"straight", "tucked"}
            kwargs (dict): Additional keyword arguments that are used for other super() calls from subclasses, allowing
                for flexible compositions of various object subclasses (e.g.: Robot is USDObject + ControllableObject).
        """
        # Store args
        self.torso_dof = torso_dof
        self.enable_torso = enable_torso
        self.default_arm_pose = default_arm_pose

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
            **kwargs,
        )


    @property
    def model_name(self):
        return "CURI"

    @property
    def discrete_action_list(self):
        # Not supported for this robot
        raise NotImplementedError()

    def _create_discrete_action_space(self):
        # Fetch does not support discrete actions
        raise ValueError("CURI does not support discrete actions!")

    def _postprocess_control(self, control, control_type):
        # Run super method first
        u_vec, u_type_vec = super()._postprocess_control(control=control, control_type=control_type)

        # Override torso value if we're keeping the torso rigid
        if self.enable_torso is False:
            u_vec[self.torso_control_idx] = self._default_torso_joint_pos
            u_type_vec[self.torso_control_idx] = ControlType.POSITION

        # Return control
        return u_vec, u_type_vec

    def _get_proprioception_dict(self):
        dic = super()._get_proprioception_dict()

        # Add torso info
        joint_positions = self.get_joint_positions(normalized=False)
        joint_velocities = self.get_joint_velocities(normalized=False)
        dic["torso_qpos"] = joint_positions[self.torso_control_idx]
        dic["torso_qvel"] = joint_velocities[self.torso_control_idx]

        return dic

    @property
    def default_proprio_obs(self):
        obs_keys = super().default_proprio_obs
        return obs_keys + ["torso_qpos"]
    
    @property
    def n_arms(self):
        return 2
    
    @property
    def arm_names(self):
        # return []
        return ["left", "right"]
    
    def update_controller_mode(self):
        super().update_controller_mode()
        # overwrite joint params (e.g. damping, stiffess, max_effort) here

        # set base joint properties
        # for joint_name in self.base_joint_names:
        #     self.joints[joint_name].stiffness = m.BASE_JOINT_STIFFNESS
        #     self.joints[joint_name].damping = m.BASE_JOINT_DAMPING
        #     self.joints[joint_name].max_effort = m.BASE_JOINT_MAX_EFFORT

        # set torso joint properties
        for joint_name in self.torso_joint_names:
            # self.joints[joint_name].stiffness = m.TORSO_JOINT_STIFFNESS
            # self.joints[joint_name].damping = m.TORSO_JOINT_DAMPING
            # self.joints[joint_name].max_effort = m.TORSO_JOINT_MAX_EFFORT
            # self.joints[joint_name].max_velocity = m.TORSO_JOINT_MAX_VELOCITY
            print(self.joints[joint_name].stiffness)
            print(self.joints[joint_name].damping)
            print(self.joints[joint_name].max_effort)

        # set arm joint properties
        # for arm in self.arm_joint_names:
        #     for joint_name in self.arm_joint_names[arm]:
        #         # self.joints[joint_name].damping = m.ARM_JOINT_DAMPING
        #         self.joints[joint_name].stiffness = 0.1* self.joints[joint_name].stiffness  # m.ARM_JOINT_STIFFNESS
        #         self.joints[joint_name].max_effort = 0.1* self.joints[joint_name].max_effort  # m.ARM_JOINT_MAX_EFFORT
        #         print(joint_name)
        #         print(self.joints[joint_name].stiffness)
        #         print(self.joints[joint_name].damping)
        #         print(self.joints[joint_name].max_effort)
        # # set finger joint properties
        # for arm in self.finger_joint_names:
        #     for joint_name in self.finger_joint_names[arm]:
        #         self.joints[joint_name].stiffness = 0.1* self.joints[joint_name].stiffness  # m.FINGER_JOINT_STIFFNESS
        #         self.joints[joint_name].max_effort = 0.1* self.joints[joint_name].max_effort  # m.FINGER_JOINT_MAX_EFFORT
        # import sys
        # sys.exit()
    @property
    def controller_order(self):
        # Ordered by general robot kinematics chain
        # the control of torso is merged into the control of default arm
        controllers = ["base", "camera"]
        for arm in self.arm_names:
            controllers += ["arm_{}".format(arm), "gripper_{}".format(arm)]

        return controllers
    
    @property
    def _default_controllers(self):
        # Always call super first
        controllers = super()._default_controllers
        
        # We use joint controllers for base and camera(head) as default
        controllers["base"] = "DifferentialDrive4WheelController"  # "DifferentialDrive4WheelController"
        # the control of torso is merged into the control of default arm
        # controllers["torso"] = "JointController"
        controllers["camera"] = "JointController"  # "JointController"
        # We use multi finger gripper, and IK controllers for eefs as default
        for arm in self.arm_names:
            controllers["arm_{}".format(arm)] = "InverseKinematicsController"  # "InverseKinematicsController"
            
            controllers["gripper_{}".format(arm)] = "JointController"  # "MultiFingerGripperController"

        return controllers
    
    @property
    def default_arm(self):
        """
        Returns:
            str: Default arm name for this robot, corresponds to the first entry in @arm_names by default
        """
        return "right"

    @property
    def wheel_radius(self):
        return 0.038

    @property
    def wheel_axle_length(self):
        return 0.230
    
    @property
    def control_limits(self):
        # Overwrite the control limits with the maximum linear and angular velocities for the purpose of clip_control
        # Note that when clip_control happens, the control is still in the base_footprint_link ("base_footprint") frame
        # Omniverse still thinks these joints have no limits because when the control is transformed to the root_link
        # ("base_footprint_x") frame, it can go above this limit.
        limits = super().control_limits
        limits["velocity"][0][self.base_control_idx] = -m.MAX_LINEAR_VELOCITY
        limits["velocity"][1][self.base_control_idx] = m.MAX_LINEAR_VELOCITY
        limits["velocity"][0][self.base_control_idx] = -m.MAX_ANGULAR_VELOCITY
        limits["velocity"][1][self.base_control_idx] = m.MAX_ANGULAR_VELOCITY
        return limits
    
    @property
    def _default_controller_config(self):
        # Grab defaults from super method first
        cfg = super()._default_controller_config

        cfg["base"][
            self._default_base_differential_drive_4wheel_controller_config["name"]
        ] = self._default_base_differential_drive_4wheel_controller_config

        for arm in self.arm_names:
            for arm_cfg in cfg["arm_{}".format(arm)].values():
                if arm == self.default_arm:
                    # Need to override joint idx being controlled to include torso in default arm controller configs
                    arm_control_idx = np.concatenate([self.torso_control_idx, self.arm_control_idx[arm]])
                    arm_cfg["dof_idx"] = arm_control_idx

                    # Need to modify the default joint positions also if this is a null joint controller
                    if arm_cfg["name"] == "NullJointController":
                        arm_cfg["default_command"] = self.reset_joint_pos[arm_control_idx]
                    
                    if self.enable_torso is False:
                        # If torso is fixed,  we also clamp its limits
                        arm_cfg["control_limits"]["position"][0][self.torso_control_idx] = \
                            self._default_torso_joint_pos
                        arm_cfg["control_limits"]["position"][1][self.torso_control_idx] = \
                            self._default_torso_joint_pos
                elif arm_cfg["name"] == "InverseKinematicsController":
                    arm_cfg["robot_description_path"] = self.robot_arm_descriptor_yamls[f"{arm}_fixed"]

        return cfg
    
    @property
    def _default_base_differential_drive_4wheel_controller_config(self):
        """
        Returns:
            dict: Default differential drive controller config to
                control this robot's base with 4 wheels.
        """
        return {
            "name": "DifferentialDrive4WheelController",
            "control_freq": self._control_freq,
            "wheel_radius": self.wheel_radius,
            "wheel_axle_length": self.wheel_axle_length,
            "control_limits": self.control_limits,
            "dof_idx": self.base_control_idx,
        }

    @property
    def _default_arm_ik_controller_configs(self):
        """
        Returns:
            dict: Dictionary mapping arm appendage name to default controller config for an
                Inverse kinematics controller to control this robot's arm
        """
        dic = super()._default_arm_ik_controller_configs 
        for arm in self.arm_names:
            dic[arm].update({
                "use_impedances": False
            })
        return dic

    @property
    def _default_joint_pos(self):
        pos = np.zeros(self.n_joints)
        pos[self.base_control_idx] = self._default_base_joint_pos
        pos[self.torso_control_idx] = self._default_torso_joint_pos
        pos[self.camera_control_idx] = self._default_camera_joint_pos
        for arm in self.arm_names:
            pos[self.arm_control_idx[arm]] = self._default_arm_joint_pos[arm]
            pos[self.gripper_control_idx[arm]] = self._default_gripper_joint_pos[arm]  # close gripper

        return pos

    @property
    def _default_base_joint_pos(self):
        return np.array([0.0, 0.0, 0.0, 0.0])

    @property
    def _default_torso_joint_pos(self):
        return np.zeros(self.torso_dof)
    
    @property
    def _default_camera_joint_pos(self):
        return np.array([0.0, 0.0])
    
    @property
    def _default_arm_joint_pos(self):
        return {"left": self._default_left_arm_joint_pos, "right": self._default_right_arm_joint_pos}

    @property
    def _default_left_arm_joint_pos(self):
        return np.array([0.6114786727643307, 0.6326230493749576, 0.05701975241122354, -0.5785648083962058, 0.24538500668160007, 2.573797290042913, 0.40057479712367045])
    
    @property
    def _default_right_arm_joint_pos(self):
        return np.array( [-1.0073911760397125, 0.8474641609359035, 1.5402616204020578, -0.6941912798610799, -1.8713471498933747, 2.1457223440806072, 2.180731299913705])

    @property
    def _default_gripper_joint_pos(self):
        return {arm: np.array([0.00, 0.00]) for arm in self.arm_names}
        # return {arm: np.array([0.00]) for arm in self.arm_names}


    @property
    def finger_lengths(self):
        return {arm: 0.1 for arm in self.arm_names}

    @property
    def base_control_idx(self):
        # return np.array([0, 1, 2, 3])
        joints = list(self.joints.keys())
        control_idx = []
        for joint in self.base_joint_names:
            control_idx.append(joints.index(joint))
        return np.array(control_idx)
    
    @property
    def torso_control_idx(self):
        joints = list(self.joints.keys())
        control_idx = []
        for joint in self.torso_joint_names:
            control_idx.append(joints.index(joint))
        return np.array(control_idx)

    @property
    def camera_control_idx(self):
        joints = list(self.joints.keys())
        control_idx = []
        for joint in self.camera_joint_names:
            control_idx.append(joints.index(joint))
        return np.array(control_idx)
    
    @property
    def arm_control_idx(self):
        joints = list(self.joints.keys())
        control_idx = {arm: [] for arm in self.arm_names}
        for arm in self.arm_names:
            for joint in self.arm_joint_names[arm]:
                control_idx[arm].append(joints.index(joint))
            control_idx[arm] = np.array(control_idx[arm])
        return control_idx

    @property
    def gripper_control_idx(self):
        # if self.torso_dof == 1
        #     return {"left": np.array([21, 22]), "right": np.array([23, 24])}
        joints = list(self.joints.keys())
        control_idx = {arm: [] for arm in self.arm_names}
        for arm in self.arm_names:
            for joint in self.finger_joint_names[arm]:
                control_idx[arm].append(joints.index(joint))
            control_idx[arm] = np.array(control_idx[arm])
        return control_idx


    @property
    def base_joint_names(self):
        return ["summit_xls_back_left_wheel_joint",
                "summit_xls_back_right_wheel_joint",
                "summit_xls_front_left_wheel_joint",
                "summit_xls_front_right_wheel_joint"]

    @property
    def torso_joint_names(self):
        return [f"torso_actuated_joint{i+1}" for i in range(self.torso_dof)]

    @property
    def camera_joint_names(self):
        return ["head_actuated_joint1", "head_actuated_joint2"]

    @property
    def arm_link_names(self):
        return {arm: [f"panda_{arm}_link{i}" for i in range(8)] for arm in self.arm_names}

    @property
    def arm_joint_names(self):
        return {arm: [f"panda_{arm}_joint{i+1}" for i in range(7)] for arm in self.arm_names}

    @property
    def eef_link_names(self):
        return {arm: f"panda_{arm}_hand" for arm in self.arm_names}

    @property
    def finger_link_names(self):
        return {arm: [f"panda_{arm}_leftfinger", f"panda_{arm}_rightfinger"] for arm in self.arm_names}

    @property
    def finger_joint_names(self):
        # return {arm: [f"panda_{arm}_finger_joint1"] for arm in self.arm_names}
        return {arm: [f"panda_{arm}_finger_joint1", f"panda_{arm}_finger_joint2"] for arm in self.arm_names}

    @property
    def usd_path(self):
        return os.path.join(gm.ASSET_PATH, f"models/curi/curi_torso{self.torso_dof}dof_gripper.usd")
        # return "/home/pjlab/OmniGibson/omnigibson/data/assets/models/curi/curi_torso1dof_gripper/curi_torso1dof_gripper.usd"

    @property
    def robot_arm_descriptor_yamls(self):
        return {"left": os.path.join(gm.ASSET_PATH, f"models/curi/yaml/curi_torso{self.torso_dof}dof_left_arm_descriptor.yaml"),
                "left_fixed": os.path.join(gm.ASSET_PATH, f"models/curi/yaml/curi_torso{self.torso_dof}dof_left_arm_fixed_torso_descriptor.yaml"),
                "right": os.path.join(gm.ASSET_PATH, f"models/curi/yaml/curi_torso{self.torso_dof}dof_right_arm_descriptor.yaml"),
                "right_fixed": os.path.join(gm.ASSET_PATH, f"models/curi/yaml/curi_torso{self.torso_dof}dof_right_arm_fixed_torso_descriptor.yaml"),
                "combined": os.path.join(gm.ASSET_PATH, f"models/curi/yaml/curi_torso{self.torso_dof}dof_dual_descriptor.yaml")}

    @property
    def urdf_path(self):
        return os.path.join(gm.ASSET_PATH, f"models/curi/curi_torso{self.torso_dof}dof_gripper.urdf")
    
    @property
    def eef_usd_path(self):
        return {self.default_arm: os.path.join(gm.ASSET_PATH, "models/curi/franka_panda_eef.usd")}
    
    @property
    def assisted_grasp_start_points(self):
        return {arm: [
            GraspingPoint(link_name=f"panda_{arm}_rightfinger", position=[0.0, 0.001, 0.045])
        ] for arm in self.arm_names}

    @property
    def assisted_grasp_end_points(self):
        return {arm: [
            GraspingPoint(link_name="panda_leftfinger", position=[0.0, 0.001, 0.045])
        ] for arm in self.arm_names}