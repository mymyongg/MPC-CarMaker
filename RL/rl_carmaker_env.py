import numpy as np
import gymnasium as gym
from casadi import interpolant
from scipy.interpolate import CubicSpline

from carmaker_env13 import CarMakerEnv
from environment.config import Config
from environment.vehicle_model import VehicleModel
from environment.track import Track
from mpc import MPC

class RLCarMakerEnv(gym.Env):
    """
    RL environment for training with CarMaker-Simulink.
    """

    def __init__(self, port, matlab_path, simul_path, save_data, render_mode=None):
        super(RLCarMakerEnv, self).__init__()

        self._config = Config()
        self._vehicle_model = VehicleModel(self._config)
        self._track = Track()
        self.controller = MPC(self._vehicle_model, self._track, self._config)
        self.carmaker_env = CarMakerEnv(host='127.0.0.1', port=port, matlab_path=matlab_path, simul_path=simul_path, save_data=save_data)
        
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
            shape = (3,),
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
        self.qddelta = self._config.mpc.qddelta
        self.qtheta = self._config.mpc.qtheta
        self.qc = self._config.mpc.qc
        self.border_left = interpolant("bound_l_s", "bspline", [self._track.X.tolist()], self._track.border_left.tolist())
        self.border_right = interpolant("bound_l_s", "bspline", [self._track.X.tolist()], self._track.border_right.tolist())
        self.control = np.zeros((1,))

        self._X_start = self._track.X[0]
        self._X_end = self._track.X[-1]

        self._cnt_mpc_status = 0
        self.sim_started = False
        self.before_X = np.array([0.0])
        self.dt = self._config.mpc.dt

        # Load ec reference for reward
        ec_data = np.load("environment/track/data/ec_data.npz")
        x = np.append(ec_data["x"], np.linspace(401, 501, 101))
        ec = np.append(ec_data["ec"], np.zeros((101,)))
        self.reference_ec = CubicSpline(x, ec)

        # For debugging
        self.total_reward = 0
        self.ep_len = 0

        print("Finishing Initialization")


    def reset(self, seed=None):
        print("\nStarting reset\n")
        self.vehicle_state = self.carmaker_env.reset()
        self.qddelta = self._config.mpc.qddelta
        self.qtheta = self._config.mpc.qtheta
        self.qc = self._config.mpc.qc
        self.control = np.zeros((1,))
        self.controller = MPC(self._vehicle_model, self._track, self._config)

        # Wait until MPC stabilizes
        while True:
            _, _ = self.controller.solve(self.vehicle_state,  np.array([self.qddelta, self.qtheta, self.qc]))
            if self.controller.status == 0:
                print("Sim started")
                break
            
            for _ in range(int(self.dt/0.001)):
                self.vehicle_state, _, _, _ = self.carmaker_env.step(np.array([0.0]))
            

        observation = np.append(self._get_normalized_state(), np.array([self.qddelta, self.qtheta, self.qc]))
        observation = np.append(observation, self._get_barrier().astype(np.float32))

        self._cnt_mpc_status = 0
        self.sim_started = False
        self.before_X = self.vehicle_state[0]

        self.total_reward = 0
        self.ep_len = 0

        return observation.astype(np.float32), {}


    def step(self, action):
        """
        action: [qddelta, qtheta, qc]. shape:(3,)
        """
        
        # Update q param
        self.qddelta = self.qddelta + 1e-4*action[0]
        self.qtheta = self.qtheta + 1e-8*action[1]
        self.qc = self.qc + 1e-5*action[2]

        # Init
        done = False
        simul_done = False
        info = {}

        # Solve MPC
        self.control, feasible = self.controller.solve(self.vehicle_state,  np.array([self.qddelta, self.qtheta, self.qc]))
        
        # Step with environment depending on dt
        for _ in range(int(self.dt/0.001)):
            self.vehicle_state, _, simul_done, info = self.carmaker_env.step(self.control)
            if simul_done:
                break

        # Check if MPC is feasible
        if self.controller.status != 0:
            self._cnt_mpc_status += 1
        else:
            self._cnt_mpc_status = 0
        
        # Check for done
        if self._cnt_mpc_status >= 10 or feasible:
            print("DONE: MPC issue")
            done = True
        if self.check_collision():
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
        reward = self.get_reward(feasible=feasible)
        
        self.total_reward += reward
        self.ep_len += 1
        self.before_X = self.vehicle_state[0] # should be after getting reward

        if done:
            print("\nForce quiting")
            sim_control = np.array([0.0])
            while True:
                sim_state, _, simul_done, _ = self.carmaker_env.step(sim_control)
                if sim_state[-2] > 0:
                    sim_control = np.array([-1])
                elif sim_state[-2] == 0:
                    sim_control = np.array([0.0])
                else:
                    sim_control = np.array([1])
                # print("delta:", sim_state[-2], "sim_control:", sim_control)
                
                if simul_done:
                    break

        done = done or simul_done
        if done:
            print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
            print("reward:", self.total_reward, "episode length:", self.ep_len)
            print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
        
        return observation.astype(np.float32), reward, done, False, info # - 1/100*self.cost


    def get_reward(self, feasible):
        reward = 0

        # 충돌시 penalty
        if self.check_collision():
            reward += -10

        # minimize ec
        ref_ec = self.reference_ec(self.vehicle_state[0])
        if ref_ec*self.vehicle_state[1] > 0: # 같은 부호
            reward += np.sign(ref_ec)*(ref_ec - self.vehicle_state[1])
        else: # 다른 부호
            reward += -abs(ref_ec - self.vehicle_state[1])

        # when MPC gets unfeasible
        if self._cnt_mpc_status >= 10 or feasible:
            reward += -5

        return reward
        

    def render(self):
        pass
        
        

    def check_collision(self):
        cone_position, cone_index = self._track.get_nearlest_cone(self.vehicle_state[0], return_cone_index=True)
        cone_position_rel = self._vehicle_model.R(self.vehicle_state[2]).T @ (cone_position - self.vehicle_state[:2])

        if abs(cone_position_rel[0]) < 0.5 * self._vehicle_model.L + self._config.env.cone_radius and abs(cone_position_rel[1]) < 0.5 * self._vehicle_model.W + self._config.env.cone_radius:
            self.collided_cone = (cone_position, cone_index)
            print("Collision occurred")
            return True
        return False


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