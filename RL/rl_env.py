import numpy as np
# import gym
import gymnasium as gym
from casadi import interpolant
from scipy.interpolate import CubicSpline

from environment import SlalomTest
from environment.config import Config
from environment.vehicle_model import VehicleModel
from environment.track import Track
from mpc import MPC

class RLEnv(gym.Env):
    """
    RL environment for training with 2D environment.
    """

    def __init__(self, is_render, render_mode=None):
        super(RLEnv, self).__init__()
        # self.is_render = is_render

        self._config = Config()
        self._vehicle_model = VehicleModel(self._config)
        self._track = Track()
        self.controller = MPC(self._vehicle_model, self._track, self._config)
        self.slalom_env = SlalomTest(is_render, render_mode='human')
        
        self.h = self._config.rl.h

        """
        Actions : (3,)
        |     qddelta, qtheta, qc difference    |
        -----------------------------------------
        |              shape: (3,)              |
        """
        self.action_space = gym.spaces.Box(
            low = -1.0,
            high = 1.0,
            shape = (3,), # only ddt
            dtype = np.float32
        )

        """
        Observation : (108,)
        |       Y, psi, vy, omega, delta      |        qddelta, qtheta, qc        |        barrier for length h       |
        ---------------------------------------------------------------------------------------------------------------
        |  shape: (5,), low: -inf, high: inf  | shape: (3,), low: -inf, high: inf | shape:(2, 50), low, high: y좌표제한 |
        """
        self.observation_space = gym.spaces.Box(
           low = -np.inf,
           high = np.inf,
           shape = (5+3+2*50, ),
           dtype = np.float32
           )

        self.vehicle_state = None
        self.border_left = interpolant("bound_l_s", "bspline", [self._track.X.tolist()], self._track.border_left.tolist())
        self.border_right = interpolant("bound_l_s", "bspline", [self._track.X.tolist()], self._track.border_right.tolist())

        self._X_start = self._track.X[0]
        self._X_end = self._track.X[-1]

        self._cnt_mpc_status = 0
        self.before_X = None

        ec_data = np.load("environment/track/data/ec_data.npz")
        self.reference_ec = CubicSpline(ec_data["x"], ec_data["ec"])

        self.qddelta = self._config.mpc.qddelta
        self.qtheta = self._config.mpc.qtheta
        self.qc = self._config.mpc.qc

        # Just for checking....
        self.total_reward = 0
        self.ep_len = 0

        print("Finishing Initialization")


    def reset(self, seed=None):

        # Reset param
        self.qddelta = self._config.mpc.qddelta
        self.qtheta = self._config.mpc.qtheta
        self.qc = self._config.mpc.qc
        self._cnt_mpc_status = 0
        self.before_X = None
        self.controller = MPC(self._vehicle_model, self._track, self._config)

        # Reset 2D env
        self.vehicle_state, info = self.slalom_env.reset(seed=seed, return_info=True)
        
        # Reset observation
        observation = np.append(self._get_normalized_state(), np.array([self.qddelta, self.qtheta, self.qc]))
        observation = np.append(observation, self._get_barrier().astype(np.float32))
        
        self.total_reward = 0
        self.ep_len = 0

        return observation, info


    def step(self, action):
        """
        action: [ddt, array of ecref]
        state: [X, Y, psi,]
        """

        # Update q param
        self.qddelta = self.qddelta * (1 + 1e-4*action[0])
        self.qtheta = self.qtheta * (1 + 1e-4*action[1])
        self.qc = self.qc * (1 + 1e-4*action[2])
       
        # Solve MPC
        control, feasible = self.controller.solve(self.vehicle_state, np.array([self.qddelta, self.qtheta, self.qc]))
        
        # Step with environment
        self.vehicle_state, _, done, truncated, info = self.slalom_env.step(control, dt=self._config.env.dt, trajectory=self.controller.trajectories)
        
        # Check if MPC is feasible
        if self.controller.status != 0:
            self._cnt_mpc_status += 1
        else:
            self._cnt_mpc_status = 0
        if self._cnt_mpc_status >= 10 or feasible:
            print("Done by MPC")
            done = True
        
        # Check if q array is feasible
        if self.qddelta > 1e+4 or self.qddelta < 1e-4:
            print("Done by qddelta")
            done = True
        if self.qtheta > 1e+0 or self.qtheta < 1e-8:
            print("Done by qtheta")
            done = True
        if self.qc > 1e+3 or self.qc < 1e-5:
            print("Done by qc")
            done = True

        # Get observation
        observation = np.append(self._get_normalized_state(), np.array([self.qddelta, self.qtheta, self.qc]))
        observation = np.append(observation, self._get_barrier().astype(np.float32))
        
        # Get reward
        reward = self.get_reward(done, truncated, feasible)

        # For logging
        self.total_reward += reward
        self.ep_len += 1

        self.before_X = self.vehicle_state[0]
        # print(reward)

        if done or truncated:
            print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
            print("reward:", self.total_reward, "episode length:", self.ep_len)
            print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")

        return observation, reward, done, truncated, info


    def get_reward(self, done, truncated, feasible):
        reward = 0

        # 충돌시 penalty
        if self.slalom_env.collided_cone is not None:
            reward += -10

        # minimize ec
        reward += abs(self.reference_ec(self.vehicle_state[0])) - abs(self.vehicle_state[1])

        # when MPC gets unfeasible
        if self._cnt_mpc_status >= 10 or feasible:
            reward += -5
            # print("feasible:", -5)

        return reward
        

    def render(self):
        self.slalom_env.render()
        

    def _get_barrier(self):
        X = self.vehicle_state[0]
        barrier = np.ndarray((2, 50))

        for i in range(int(self.h / 2)):
            if X + self.h > self._X_end:
                barrier[0, i] = self.border_left(self._X_end)
                barrier[1, i] = self.border_right(self._X_end)
            else:
                barrier[0, i] = self.border_left(X + 2*i)
                barrier[1, i] = self.border_right(X + 2*i)
                
        return barrier
    
    def _get_normalized_state(self): 
        # X, Y, psi, vx, vy, omega, delta, tau

        Y = self.vehicle_state[1]
        psi = self.vehicle_state[2]
        vy = self.vehicle_state[4]
        omega = self.vehicle_state[5]
        delta = self.vehicle_state[-2]

        return np.array([Y, psi, vy, omega, delta], dtype=np.float32)