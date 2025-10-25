from os import path
import numpy as np
import gymnasium as gym
import mujoco
import mujoco.viewer
import glfw

from .transforms import rpy2r, r2rpy,r2w
from .utils import (
    trim_scale,
    compute_view_params,
    get_idxs,
    get_colors,
    get_monitor_size,
    TicTocClass,
)

class SO100Env(gym.Env):
    def __init__(self, model_path="so100_scene.xml",render_mode=None,
                 camera_res=(800, 600)):
        self.render_mode = render_mode
        self.camera_res=camera_res
        if model_path.startswith(".") or model_path.startswith("/"):
            self.model_path = model_path
        elif model_path.startswith("~"):
            self.model_path = path.expanduser(model_path)
        else:
            self.model_path = path.join(path.dirname(__file__), "assets", model_path)
        if not path.exists(self.model_path):
            raise OSError(f"File {self.model_path} does not exist")
        self.model=mujoco.MjModel.from_xml_path(self.model_path)
        self.data=mujoco.MjData(self.model)
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

        # self.joint_names = [mujoco.mj_id2name(self.model,mujoco.mjtObj.mjOBJ_JOINT,joint_idx)
        #                          for joint_idx in range(self.model.njnt)]
        self.joint_names = ['joint1',
                    'joint2',
                    'joint3',
                    'joint4',
                    'joint5',
                    'joint6',]
        
        self.observation_space =  gym.spaces.Dict(
            {
                "camera1":  gym.spaces.Box(0, 255, shape=(camera_res[0],camera_res[1],3), dtype=int),
                "camera2":  gym.spaces.Box(0, 255, shape=(camera_res[0],camera_res[1],3), dtype=int),
                "joints":gym.spaces.Box( low=-np.inf, high=np.inf, shape=(self.model.nq,), dtype=np.float64 )
            }
        )
        
        self.action_space = gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(7,), dtype=np.float64
            )
        
    
    def get_fixed_cam_rgb(self,cam_name):
        camera_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
        cam = mujoco.MjvCamera()
        cam.fixedcamid =camera_id
        cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        viewport = mujoco.MjrRect(0,0,self.camera_res[0],self.camera_res[1]) 
        scn  = mujoco.MjvScene(self.model, maxgeom=10000)
        context  = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150.value)
        # 更新场景
        mujoco.mjv_updateScene(self.model, self.data, mujoco.MjvOption(), mujoco.MjvPerturb(), cam, mujoco.mjtCatBit.mjCAT_ALL, scn)
        mujoco.mjr_render(viewport, scn,context )
        # 提取 RGB 图像
        rgb = np.zeros((viewport.height,viewport.width,3),dtype=np.uint8)
        depth_raw = np.zeros((viewport.height,viewport.width),dtype=np.float32)
        mujoco.mjr_readPixels(rgb,depth_raw,viewport,context )
        rgb,depth_raw = np.flipud(rgb),np.flipud(depth_raw)
        return rgb
    def _get_obs(self):
        # 初始化 OpenGL 上下文（离屏渲染）
        glfw.init()
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        window = glfw.create_window(self.camera_res[0], self.camera_res[1], "Offscreen", None, None)
        glfw.make_context_current(window)

        return {
            "camera1": self.get_fixed_cam_rgb("egocentric"), 
            "camera2": self.get_fixed_cam_rgb("egocentric"), 
            "joints":self.data.qpos.copy()
            }
    
    def get_p_body(self,body_name):
        """
        Get the position of the specified body.
        
        Parameters:
            body_name (str): Name of the body.
        
        Returns:
            np.array: The position of the body.
        """
        return self.data.body(body_name).xpos.copy()
    
    def get_R_body(self,body_name):
        """
        Get the rotation matrix of the specified body.
        
        Parameters:
            body_name (str): Name of the body.
        
        Returns:
            np.array: The 3x3 rotation matrix.
        """
        return self.data.body(body_name).xmat.reshape([3,3]).copy()
    
    def get_pR_body(self,body_name):
        """
        Get both the position and rotation matrix of the specified body.
        
        Parameters:
            body_name (str): Name of the body.
        
        Returns:
            tuple: (position, rotation matrix)
        """
        p = self.get_p_body(body_name)
        R = self.get_R_body(body_name)
        return p,R
    
    def get_ee_pose(self):
        '''
        get the end effector pose of the robot + gripper state
        '''
        p, R = self.get_pR_body(body_name='tcp_link')
        rpy = r2rpy(R)
        return np.concatenate([p, rpy],dtype=np.float32)
    
    def _get_info(self):
        return {
            "ee_pose": self.get_ee_pose()
        }
    def reset(self, seed=None, options=None):
        mujoco.mj_resetData(self.model,self.data)
        for _ in range(10):
            mujoco.mj_step(self.model,self.data)

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self.render()

        return observation, info
     
    # Inverse kinematics
    def get_J_body(self,body_name):
        """
        Compute the Jacobian matrices (position and rotation) for the specified body.
        
        Parameters:
            body_name (str): Name of the body.
        
        Returns:
            tuple: (J_p, J_R, J_full) where J_full is the stacked Jacobian.
        """
        n_dof = self.model.nv # degree of freedom (=number of columns of Jacobian)
        J_p = np.zeros((3,n_dof)) # nv: nDoF
        J_R = np.zeros((3,n_dof))
        mujoco.mj_jacBody(self.model,self.data,J_p,J_R,self.data.body(body_name).id)
        J_full = np.array(np.vstack([J_p,J_R]))
        return J_p,J_R,J_full
    
    def get_ik_ingredients(
        self,
        body_name ,
        p_trgt    = None,
        R_trgt    = None
        ):
        """
        Compute the Jacobian and error vector needed for inverse kinematics.
        
        Parameters:
            body_name (str): Name of the body (if provided).
            p_trgt (np.array): Target position.
            R_trgt (np.array): Target rotation matrix.
        
        Returns:
            tuple: (J, err) where J is the Jacobian and err is the error vector.
        """
        IK_P      = True,
        IK_R      = True,
        if p_trgt is None: IK_P = False
        if R_trgt is None: IK_R = False

        J_p,J_R,J_full = self.get_J_body(body_name=body_name)
        p_curr,R_curr = self.get_pR_body(body_name=body_name)
       
        if (IK_P and IK_R):
            p_err = (p_trgt-p_curr)
            R_err = np.linalg.solve(R_curr,R_trgt)
            w_err = R_curr @ r2w(R_err)
            J     = J_full
            err   = np.concatenate((p_err,w_err))
        elif (IK_P and not IK_R):
            p_err = (p_trgt-p_curr)
            J     = J_p
            err   = p_err
        elif (not IK_P and IK_R):
            R_err = np.linalg.solve(R_curr,R_trgt)
            w_err = R_curr @ r2w(R_err)
            J     = J_R
            err   = w_err
        else:
            J   = None
            err = None
        return J,err
    
    def damped_ls(self,J,err,eps=1e-6,stepsize=1.0,th=5*np.pi/180.0):
        """
        Solve the inverse kinematics using the damped least squares method.
        
        Parameters:
            J (np.array): Jacobian matrix.
            err (np.array): Error vector.
            eps (float): Damping factor.
            stepsize (float): Step size multiplier.
            th (float): Threshold for scaling the result.
        
        Returns:
            np.array: The computed joint increments (dq).
        """
        dq = stepsize*np.linalg.solve(a=(J.T@J)+eps*np.eye(J.shape[1]),b=J.T@err)
        dq = trim_scale(x=dq,th=th)
        return dq
    
    def get_idxs_fwd(self,joint_names):
        """
        Get the indices of joints used for forward kinematics based on joint names.
        
        Parameters:
            joint_names (list): List of joint names.
        
        Returns:
            list: Indices corresponding to the joints.

        Example:
            env.forward(q=q,joint_idxs=idxs_fwd) # <= HERE
        """
        return [self.model.joint(jname).qposadr[0] for jname in joint_names]
    
    def get_idxs_jac(self,joint_names):
        """ 
        Get the indices of joints for Jacobian calculation based on joint names.
        
        Parameters:
            joint_names (list): List of joint names.
        
        Returns:
            list: Indices corresponding to the joints.
        """
        return [self.model.joint(jname).dofadr[0] for jname in joint_names]
    
    def get_qpos_joint(self,joint_name):
        """
        Get the position for a specific joint.
        
        Parameters:
            joint_name (str): Name of the joint.
        
        Returns:
            np.array: The joint position.
        """
        addr = self.model.joint(joint_name).qposadr[0]
        L = len(self.model.joint(joint_name).qpos0)
        qpos = self.data.qpos[addr:addr+L]
        return qpos
    
    def get_qpos_joints(self,joint_names):
        """
        Get the positions for multiple joints.
        
        Parameters:
            joint_names (list): List of joint names.
        
        Returns:
            np.array: Joint positions.
        """
        return np.array([self.get_qpos_joint(joint_name) for joint_name in joint_names]).squeeze()
    
    def solve_ik(
        self,
        body_name_trgt,
        target_p          = None,
        target_r          = None,
        max_ik_tick     = 1000,
        ik_err_th       = 1e-2
    ):
        joint_idxs_jac=self.get_idxs_jac(self.joint_names)

        q_curr = self.get_qpos_joints(joint_names=self.joint_names)
        for ik_tick in range(max_ik_tick):
            J,ik_err_stack = self.get_ik_ingredients(
                body_name = body_name_trgt,
                p_trgt    = target_p,
                R_trgt    = target_r,
            )   
            delta_qpos = self.damped_ls(J,ik_err_stack,stepsize=50)
            
            # print("delta_qpos",delta_qpos)
            # print("delta_qpos[joint_idxs_jac]",delta_qpos[joint_idxs_jac])
            q_curr = q_curr + delta_qpos[joint_idxs_jac]

            joint_idxs = self.get_idxs_fwd(joint_names=self.joint_names)
            self.data.qpos[joint_idxs] = q_curr
            mujoco.mj_forward(self.model,self.data)

            ik_err = np.linalg.norm(ik_err_stack)
            # print("ik_err",ik_err)
            if ik_err < ik_err_th: 
                break # terminate condition

        if ik_err > ik_err_th:
            print ("ik_err:[%.4f] is higher than ik_err_th:[%.4f]."%
                (ik_err,ik_err_th))
            print ("You may want to increase max_ik_tick:[%d]"%
                (max_ik_tick))
            
        return q_curr
        
        
        
        
    def step(self, action):
        target_p,target_r=self.get_pR_body(body_name='tcp_link')
        target_p+=action[:3]
        target_r=target_r.dot(rpy2r(action[3:6]))
        # target_r=rpy2r(np.deg2rad([90,-0.,90 ]))
        qpos=self.solve_ik(
            body_name_trgt="tcp_link",
            target_p    = target_p,
            target_r    = target_r
        )

        gripper_cmd = np.array([action[-1]]*4)
        gripper_cmd[[1,3]] *= 0.8
        self.data.ctrl = np.concatenate([qpos[:6], gripper_cmd])
        mujoco.mj_step(self.model, self.data)

        if self.render_mode == "human":
            self.render()

        observation = self._get_obs()
        info = self._get_info()
        reward=0
        terminated=False

        return observation, reward, terminated, False, info
     
    def render(self):
        self.viewer.sync()
     
    def close(self):
        if self.viewer is not None:
            self.viewer.close()