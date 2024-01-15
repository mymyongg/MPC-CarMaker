from numpy import array, zeros, hstack, sin, cos, arctan2
from numpy.random import rand
from .acados_ocp_pp import acados_ocp_pp
from .acados_solver import acados_solver
from scipy.interpolate import CubicSpline
import numpy as np

class MPC:

    def __init__(self, vehicle_model, track, config):

        acados_ocp = acados_ocp_pp(vehicle_model, track, config)
        self.__solver = acados_solver(acados_ocp, config)

        self.track = track
        self.N = config.mpc.N
        # self.dt = config.mpc.dt
        self.nu = acados_ocp.model.u.size1()
        self.nx = acados_ocp.model.x.size1()

        self.trajectories = None
        self.theta = None

        self.qtheta = config.mpc.qtheta
        self.qc = config.mpc.qc
        self.qddelta = config.mpc.qddelta

        self.ddelta_before = 0

    def solve(self, state, q_array):  
        """
        Solve MPC given state, changed parameters.
        Arguments.
        -----------------------------------------
        state: [X, Y, psi, vx, vy, omega, delta, D]
        qarray: [qddelta, qtheta, qc]
        """
        
        # Compute the state to path-parametric state
        theta0 = state[0]   # X
        ec0 = state[1]    # -Y
        epsi0 = state[2]    # psi

        # Set initial state constraint
        x0 = hstack([theta0, ec0, epsi0, state[3:]])
        self.__solver.set(0, "lbx", x0)
        self.__solver.set(0, "ubx", x0)

        if self.theta is None:
            for k in range(self.N):
                self.__solver.set(k, "x", x0)

        for k in range(self.N):
            self.__solver.set(k, "p", hstack([self.ddelta_before, q_array[0], q_array[1], theta0, q_array[2]]))
        
        self.status = self.__solver.solve()

        
        if self.trajectories is None:
            self.trajectories = [zeros((self.N, self.nx))]
        # self.control = u0

        # MPC trajectory
        for k in range(self.N):
            self.trajectories[0][k, :] = self.__solver.get(k, "x")


        if self.status==0:
            self.theta = self.__solver.get(1, "x")[0]

        # Check if the caculated track has gone off road
        feasible = False
        for k in range(self.N):
            if self.__solver.get(k, "x")[1] < -5 or self.__solver.get(k, "x")[1] > 5:
                print("Prediction gone off road")
                feasible = True
                break
       
        # Get MPC results
        x0 = self.__solver.get(0, "x")
        u0 = self.__solver.get(0, "u")
        self.ddelta_before = u0[0]

        return u0, feasible
    

    def reset_theta(self):
        self.theta = None
