import numpy as np
import gym

class Frequency(gym.Env):
    def __init__(self, M, D, Pm, max_action, dim_action, delta_t, Penalty_action, F):
        super(Frequency, self).__init__()

        self.step_count = 0
        self.done = False
        '''Parameters'''
        self.param_gamma = 1                    # decay parameter for rewards
        self.M = M                              # Inertia constant (vector for multi-machine systems)
        self.D = D                              # Damping factor (vector for multi-machine systems)
        self.Pm = Pm                            # Net power injection
        self.max_action = max_action            # Maximum control action allowed
        self.dim_action = dim_action            # Dimension of the action space
        self.omega_scale = 2*np.pi
        self.delta_t = delta_t                  # Time step for simulation
        self.Penalty_action = Penalty_action    # Penalty term related to control actions
        self.state = []                         # State Dynamics

        # \delta is rad, \omega is multiplied with 2pi here to be converted to Hz
        self.state_transfer1 = np.vstack((np.hstack((np.identity(dim_action, dtype=np.float32), np.zeros((dim_action, dim_action), dtype=np.float32))),\
                                          np.hstack((delta_t*self.omega_scale*np.identity(dim_action, dtype=np.float32),\
                                                     np.identity(dim_action, dtype=np.float32) - delta_t*np.diag(D/M)))))

        self.state_transferF = -delta_t*(((M**(-1)).reshape(dim_action, 1)) @ np.ones((1, dim_action), dtype=np.float32)) * F
        self.state_transfer2 = np.hstack((np.zeros((dim_action, dim_action), dtype=np.float32),\
                                          np.identity(dim_action, dtype=np.float32)))

        self.state_transfer3 = np.hstack((np.zeros((1, dim_action), dtype=np.float32),\
                                          delta_t*Pm*(M**(-1))))
        self.state_transfer3_Pm = np.hstack((np.zeros((dim_action, dim_action), dtype=np.float32),\
                                             delta_t*np.diag((M**(-1)))))
        self.state_transfer4 = np.hstack((np.zeros((dim_action, dim_action), dtype=np.float32),\
                                          -delta_t*np.diag((M**(-1)))))

        self.select_add_w = np.vstack((np.zeros((dim_action, 1), dtype=np.float32),\
                                       np.ones((dim_action, 1), dtype=np.float32)))
        self.select_w = np.vstack((np.zeros((dim_action, dim_action), dtype=np.float32),\
                                   np.identity(dim_action, dtype=np.float32)))
        self.select_delta = np.vstack((np.identity(dim_action, dtype=np.float32),\
                                       np.zeros((dim_action, dim_action), dtype=np.float32)))

    def step(self, action, Pm_change):
        # Simulate one time step of the environment
        if self.state.ndim == 1:
            self.state = self.state.reshape(1, -1)
            
        self.state = self.state @ self.state_transfer1 +\
              np.sum(np.sin(np.transpose(self.state @ self.select_delta) @ np.ones((1, self.dim_action), dtype=np.float32) -\
                             np.ones((self.dim_action, 1), dtype=np.float32) @ (self.state@self.select_delta)) * self.state_transferF, axis=1) @ self.state_transfer2 +\
                             Pm_change @ self.state_transfer3_Pm +\
                             action @ self.state_transfer4
        loss_freq = - self.param_gamma * abs(self.state) @ self.select_add_w
        # loss_action = - self.Penalty_action * pow(action, 2).sum()
        # loss = loss_freq + loss_action
        loss = loss_freq

        self.step_count += 1
        if self.step_count >= 400:
            self.done = True

        return self.state, loss, self.done

    def set_state(self, state_input):
        # Set the state to a specific input value
        self.state = state_input

    def reset(self):
        # Initialize the state with random values within a specific range
        self.done = False
        self.step_count = 0
        initial_state1 = np.random.uniform(-0.1, 0.1, (1, self.dim_action))
        initial_state2 = np.random.uniform(-0.03, 0.03, (1, self.dim_action))
        s_concate = np.hstack((initial_state1, initial_state2)).astype(np.float32)
        self.state = s_concate

        return self.state