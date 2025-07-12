import matplotlib.pyplot as plt
import numpy as np
from mat4py import loadmat
import torch
import time

from env.interact_env import Frequency
from piecewise_linear.train import MonotonicRNN, plot_loss_record_from_file, plot_u_from_weights

'''Data of Kron IEEE 39-bus system'''
data = loadmat('../data/IEEE_39bus_Kron.mat')
K_EN = data['Kron_39bus']['K']      # Admittance matrix
K_EN = np.asarray(K_EN, dtype=np.float32)
H = data['Kron_39bus']['H']         # Inertia constants
H = np.asarray(H, dtype=np.float32)
omega_R = data['Kron_39bus']['omega_R']     # Reference rotational speed

'''Parameters for interactive environment'''
action_dim = 10
state_dim = 2*action_dim
delta_t = 0.01
M = H.reshape(action_dim)*2/omega_R*2*np.pi     # inertia constant
# Here use the Damping coefficient data from P. Demetriou, M. Asprou, et al. "Dynamic IEEE test systems for transient analysis"
D = np.zeros(action_dim, dtype=np.float32)
D[0] = 2*590/100        # Damping coefficient of generator 1
D[1:8] = 2*865/100      # Damping coefficient of generator 2 to 8
D[8:10] = 2*911/100     # Damping coefficient of generator 9 and 10
D = D/omega_R*2*np.pi 
F = K_EN                # admittance matrix
Penalty_action = 0.1      # penalty coefficient to restrict control input
# Net power injection of each bus (unit: p.u.)
Pi = np.array([[-0.19983394, -0.25653884, -0.25191885, -0.10242008, -0.34510365,
         0.23206371,  0.4404325 ,  0.5896664 ,  0.26257738, -0.36892463]], dtype=np.float32)
# max_action, which represents the control input range of renewable energy
max_action = np.array([[0.19606592, 0.2190382 , 0.22375287, 0.0975513 , 0.29071101,
        0.22091283, 0.38759459, 0.56512538, 0.24151538, 0.29821917]], dtype=np.float32)
# equilibrium state of the system
equilibrium_init = np.array([[ -0.05420687, -0.07780334, -0.07351729, -0.05827823, -0.09359571,
        -0.02447385, -0.00783582,  0.00259523, -0.0162409 , -0.06477749,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.       ]], dtype=np.float32)

'''Interactive environment initialization'''
env_name = 'IEEE39'
env = Frequency(M, D, Pi, max_action, action_dim, delta_t, \
                Penalty_action, F)
env.reset()

'''Hyperparameters for training'''
state_units = action_dim*2
action_units = action_dim
internal_units = 64
batch_size = 800
episodes = 1000
num_gen_step = 3
step_magnitude = 1
T = 200
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

trainer = MonotonicRNN(env, state_units, action_units, internal_units, batch_size,\
                       episodes, T, equilibrium_init, num_gen_step, step_magnitude, device)

start_time = time.time()
trainer.train(Pi)
end_time = time.time()
print("Training time: ", end_time - start_time)

model_name = "Monotonic_controller.pth"
result_name = "Monotonic_RNN_resutls.png"
policy_name = "Learned_policy.png"
weight_name = "controller_weights.npz"
trainer.save(model_name)
trainer.export_controller_weights(weight_name)
trainer.plot_loss_record()

'''Parameters for visualization'''
SimulationLength = 1000
gen_id = [2, 3, 8]

trainer.result_vis(model_name, result_name, Pi, gen_id, SimulationLength)
trainer.plot_learned_policies(model_name, policy_name)
# plot_loss_record_from_file()
# plot_u_from_weights(weight_name)