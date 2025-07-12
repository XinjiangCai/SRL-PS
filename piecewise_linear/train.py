import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

class CustomRNNCell(nn.Module):
    def __init__(self, env, action_units, internal_units):
        super().__init__()
        self.action_units = action_units
        self.internal_units = internal_units

        self.register_buffer("state_transfer1", \
                             torch.tensor(env.state_transfer1, dtype=torch.float32))
        self.register_buffer("state_transferF", \
                             torch.tensor(env.state_transferF, dtype=torch.float32))
        self.register_buffer("state_transfer2", \
                             torch.tensor(env.state_transfer2, dtype=torch.float32))
        self.register_buffer("state_transfer3", \
                             torch.tensor(env.state_transfer3, dtype=torch.float32))
        self.register_buffer("state_transfer4", \
                             torch.tensor(env.state_transfer4, dtype=torch.float32))
        self.register_buffer("state_transfer3_Pm", \
                             torch.tensor(env.state_transfer3_Pm, dtype=torch.float32))

        self.register_buffer("select_add_w", \
                             torch.tensor(env.select_add_w, dtype=torch.float32))
        self.register_buffer("select_w", \
                             torch.tensor(env.select_w, dtype=torch.float32))
        self.register_buffer("select_delta", \
                             torch.tensor(env.select_delta, dtype=torch.float32))
        self.register_buffer("max_action", \
                             torch.tensor(env.max_action, dtype=torch.float32))

        self.register_buffer("w_recover", torch.eye(internal_units) - \
                             torch.diag(torch.ones(internal_units - 1), diagonal=1))
        self.register_buffer("b_recover", \
                             torch.triu(torch.ones((internal_units, internal_units)), diagonal=1))
        self.register_buffer("ones_frequency", \
                             torch.ones((action_units, internal_units), dtype=torch.float32))

        self.w_plus_temp0 = nn.Parameter(torch.rand(action_units, internal_units) * 0.1)
        self.b_plus_temp0 = nn.Parameter(torch.rand(action_units, internal_units) * 0.1)
        self.w_minus_temp0 = nn.Parameter(torch.rand(action_units, internal_units) * 0.1)
        self.b_minus_temp0 = nn.Parameter(torch.rand(action_units, internal_units) * 0.1)

    def forward(self, inputs, prev_output):
        if len(prev_output.shape) == 1:
            prev_output = prev_output.unsqueeze(0)
        if len(inputs.shape) == 1:
            inputs = inputs.unsqueeze(0)
        
        batch_size = prev_output.size(0)
        Multiply_ones = torch.ones((batch_size, self.action_units, self.action_units), \
                                   device = prev_output.device)
        
        # Build effective weights with structure and positivity
        w_plus_temp = self.w_plus_temp0 ** 2
        b_plus_temp = self.b_plus_temp0 ** 2
        w_minus_temp = self.w_minus_temp0 ** 2
        b_minus_temp = self.b_minus_temp0 ** 2

        w_plus = torch.matmul(w_plus_temp, self.w_recover).unsqueeze(0)
        b_plus = torch.matmul(-b_plus_temp, self.b_recover).unsqueeze(0)
        w_minus = torch.matmul(-w_minus_temp, self.w_recover).unsqueeze(0)
        b_minus = torch.matmul(-b_minus_temp, self.b_recover).unsqueeze(0)

        diag_w = torch.diag_embed(prev_output @ self.select_w)
        linear_input = torch.matmul(diag_w, self.ones_frequency).squeeze()
        diag_delta = torch.diag_embed(prev_output @ self.select_delta)

        # Control actions (batch matrix multiplication)
        nonlinear_plus = torch.sum(F.relu(linear_input + \
                                          b_plus) * w_plus, dim=2)
        nonlinear_minus = torch.sum(F.relu(-linear_input + \
                                           b_minus) * w_minus, dim=2)
        action_nonconstrained = nonlinear_plus + nonlinear_minus

        max_action = self.max_action
        action = max_action - F.relu(max_action - action_nonconstrained) +\
              F.relu(-max_action - action_nonconstrained)

        # New state computation
        delta_term = torch.sum(
             torch.sin(torch.matmul(diag_delta, torch.ones((self.action_units, self.action_units), \
                                                           device = prev_output.device)) - \
                       torch.matmul(Multiply_ones, torch.diag_embed(prev_output @ self.select_delta))) * \
                        self.state_transferF, dim=2
        )
        new_state = prev_output @ self.state_transfer1 + delta_term @ self.state_transfer2 + \
                    self.state_transfer3 + action @ self.state_transfer4 + \
                        inputs @ self.state_transfer3_Pm

        # loss0 = torch.matmul(torch.abs(new_state), self.select_add_w)
        loss0 = torch.matmul(pow(new_state, 2), self.select_add_w)
        frequency = new_state @ self.select_w

        return loss0, frequency, action, new_state.squeeze(0)

class MonotonicRNN:
    def __init__(self, env, state_units, action_units, internal_units, batch_size,\
                 episodes, T, equilibrium_init, num_gen_step, step_magnitude, device):
        self.env = env
        self.state_units = state_units
        self.action_units = action_units
        self.internal_units = internal_units
        self.batch_size = batch_size
        self.episodes = episodes
        self.T = T
        self.device = device
        self.equilibrium_init = torch.tensor(equilibrium_init, dtype=torch.float32).to(device)
        self.num_gen_step = num_gen_step
        self.step_magnitude = step_magnitude        

        self.model = CustomRNNCell(env, action_units, internal_units).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.05)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.9)
        self.loss_record = []        

    def train(self, Pi):
        np.random.seed(42)
        
        for i in range(self. episodes):
            Pi_tensor = torch.tensor(Pi, dtype=torch.float32).unsqueeze(0)
            Pi_change = Pi_tensor.repeat(self.batch_size, self.T, 1)
            Pi_change = Pi_change.cpu().numpy()
            initial_state = self.equilibrium_init.repeat(self.batch_size, 1).to(self.device)
            for _ in range(self.num_gen_step):
                idx_gen_deviation = np.random.randint(0, self.action_units, self.batch_size)    # affected generators
                idx_batch_deviation = np.random.randint(0, self.batch_size, self.batch_size)    # affected batches
                slot_start_deviation = np.random.randint(0, self.T/2, self.batch_size)          # time step when disturbance start
                step_change = np.random.uniform(-1, 1, self.batch_size) * self.step_magnitude        # magnitude of the disturbance
                # step_change = 0

                for t in range(self.T):
                    for batch in range(self.batch_size):
                        if t >= slot_start_deviation[batch]:
                            Pi_change[idx_batch_deviation[batch], t, idx_gen_deviation[batch]] += step_change[batch]

            state = initial_state
            loss0_all, freq_all, action_all = [], [], []

            for t in range(self.T):
                inputs = torch.tensor(Pi_change[:, t, :], dtype=torch.float32).to(self.device)
                loss0, frequency, action, state = self.model(inputs, state)

                loss0_all.append(loss0)
                freq_all.append(frequency)
                action_all.append(action)

            # Accumulate cost over time
            loss0_all = torch.stack(loss0_all, dim=1)
            freq_all = torch.stack(freq_all, dim=1)
            action_all = torch.stack(action_all, dim=1)
            loss_freq = torch.sum(torch.max(torch.abs(freq_all), dim=1).values) / self.batch_size
            loss_action = self.env.Penalty_action * torch.sum(pow(freq_all, 2)) / self.batch_size
            loss_freq = torch.sum(pow(freq_all, 2)) / self.batch_size
            loss = loss_action + loss_freq
            self.loss_record.append(loss.item())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            print('episode', i, 'Loss', loss)
            print('episode', i, 'Loss_frequency', loss_freq)

    def export_controller_weights(self, filename):
        weights = {
            "w_plus_temp0": self.model.w_plus_temp0.detach().cpu().numpy(),
            "b_plus_temp0": self.model.b_plus_temp0.detach().cpu().numpy(),
            "w_minus_temp0": self.model.w_minus_temp0.detach().cpu().numpy(),
            "b_minus_temp0": self.model.b_minus_temp0.detach().cpu().numpy(),
            "w_recover": self.model.w_recover.detach().cpu().numpy(),
            "b_recover": self.model.b_recover.detach().cpu().numpy(),
            "max_action": self.model.max_action.detach().cpu().numpy(),
        }
        np.savez(filename, **weights)
        print(f"[✓] Controller weights saved to {filename}")

    def save(self, filepath):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss_record': self.loss_record
        }, filepath)

    def load(self, filepath):
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.loss_record = checkpoint['loss_record']
    
    def plot_loss_record(self, filename="Training_Loss.png"):
        np.save("relu_loss_record.npy", np.array(self.loss_record))
        print(f"[✓] Loss record saved to relu_loss_record.npy")          
        plt.figure()
        plt.plot(self.loss_record, label='Training Loss')

        final_loss = self.loss_record[-1]
        plt.axhline(y=final_loss, linestyle='--', color='gray',\
                    linewidth=1.2)
        plt.text(0.6*len(self.loss_record), final_loss+0.02, \
                 f"{final_loss:.4f}", color='gray', fontsize=10)
        
        plt.xlabel('Episode')
        plt.ylabel('Loss')
        plt.xlim(0, 1000)
        plt.title('Loss per Episode')
        plt.grid(True)
        plt.legend()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"[✓] Loss curve saved to {filename}")    

    def result_vis(self, filepath, figure_name, Pi, gen_id, SimulationLength):
        self.load(filepath)
        self.model.eval()
        s = self.equilibrium_init.squeeze().cpu().numpy()
        self.env.set_state(s)

        Trajectory = [s]
        Record_u = []
        Record_loss = []
        Loss_RNN = 0

        Pi_init = Pi.copy().astype(np.float32)
        Pi1 = Pi_init.copy()
        Pi2 = Pi_init.copy()
        Pi2[0][gen_id] += 0.2
        
        for step in range(SimulationLength):
            Pi_change = Pi1 if step<200 or step>700 else Pi2

            state_tensor = torch.tensor(s, dtype=torch.float32).unsqueeze(0).to(self.device)
            input_tensor = torch.tensor(Pi_change, dtype=torch.float32).unsqueeze(0).to(self.device)
            with torch.no_grad():
                _, _, action, s = self.model(input_tensor, state_tensor)
                u = action.squeeze(0).cpu().numpy()
            # print(("u shape:", u.shape))
            next_s, r, _ = self.env.step(u, Pi_change)
            # print(("next_s shape:", next_s.shape))
            s = next_s
            Loss_RNN += r

            if s.ndim > 1:
                s = s.squeeze(0)
            Trajectory.append(s)
            Record_u.append(u)
            Record_loss.append(np.squeeze(r))

        np.save(f"relu_total_deviation.npy", Record_loss)

        print("Total RNN Loss:", Loss_RNN)
        max_step_loss = np.max(np.abs(Record_loss))
        print("Maximum Instantaneous RNN Loss:", max_step_loss)    
        Trajectory = np.squeeze(np.asarray(Trajectory))
        Record_u = np.asarray(Record_u)

        frequencies = Trajectory[:, self.action_units:]
        max_freq_devs = np.max(frequencies, axis=0)
        min_freq_devs = np.min(frequencies, axis=0)
        for i in range(self.action_units):
            print(f"$\omega_{{{i+1}}}$: Max = {max_freq_devs[i]:.6f} Hz, Min = {min_freq_devs[i]:.6f} Hz")

        plt.rcParams.update({
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.titlesize": 14
        })

        colors = plt.cm.tab10(np.linspace(0, 1, self.action_units))
        Time_line = np.arange(1, SimulationLength + 1) * self.env.delta_t

        fig, axs = plt.subplots(2, 2, figsize=(11, 8), dpi=150)
        # fig.suptitle("Trained RNN Lyapunov Controller", fontsize=16)

        axs[0, 0].plot(Time_line, Record_loss, color='tab:red')
        axs[0, 0].set_xlabel('Time (s)')
        axs[0, 0].set_ylabel('Total frequency deviations (Hz)')
        axs[0, 0].set_xlim(0, 10)
        axs[0, 0].grid(True)
        # axs[0, 0].set_title('Total frequency deviations over Time')

        for i in range(self.action_units):
            axs[0, 1].plot(Time_line, Record_u[:, i], label=f'$u_{{{i+1}}}$', color=colors[i])
        axs[0, 1].set_xlabel('Time (s)')
        axs[0, 1].set_ylabel('Control Input')
        # axs[0, 1].set_title('Control Actions')
        axs[0, 1].set_xlim(0, 10)
        axs[0, 1].legend(loc='center left', bbox_to_anchor=(1.05, 0.5), ncol=1)
        axs[0, 1].grid(True)

        Time_line_full = np.arange(1, SimulationLength + 2) * self.env.delta_t
        for i in range(self.action_units):
            axs[1, 0].plot(Time_line_full, Trajectory[:, i], label=f'$\\delta_{{{i+1}}}$', color=colors[i])
        axs[1, 0].set_xlabel('Time (s)')
        axs[1, 0].set_ylabel('Angle (rad)')
        # axs[1, 0].set_title('Phase Angle Deviations')
        axs[1, 0].set_xlim(0, 10)
        axs[1, 0].legend(loc='center left', bbox_to_anchor=(1.05, 0.5), ncol=1)
        axs[1, 0].grid(True)

        for i in range(self.action_units):
            axs[1, 1].plot(Time_line_full, Trajectory[:, self.action_units + i], label=f'$\\omega_{{{i+1}}}$', color=colors[i])
        axs[1, 1].set_xlabel('Time (s)')
        axs[1, 1].set_ylabel('Frequency Deviations (Hz)')
        # axs[1, 1].set_title('Frequency Deviations')
        axs[1, 1].set_xlim(0, 10)
        axs[1, 1].legend(loc='center left', bbox_to_anchor=(1.05, 0.5), ncol=1)
        axs[1, 1].grid(True)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(figure_name, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[✓] Figure saved to {figure_name}")

    def plot_learned_policies(self, filepath, figure_name):
        self.load(filepath)
        self.model.eval()
        # omega_vals = np.linspace(-0.2, 0.2, 2000).reshape(-1, 1)
        omega_vals = np.linspace(-0.025, 0.025, 2000).reshape(-1, 1)
        omega_tensor = torch.tensor(omega_vals, dtype=torch.float32).to(self.device)

        plt.figure()
        colors = plt.cm.tab10(np.linspace(0, 1, self.action_units))

        for i in range(self.action_units):
            with torch.no_grad():
                out = []
                for omega in omega_tensor:
                    omega_index = self.state_units - self.action_units + i
                    dummy_prev_output = torch.zeros((1, self.state_units), dtype=torch.float32).to(self.device)
                    dummy_prev_output[0, omega_index] = omega
                    dummy_input = torch.zeros((1, self.action_units), dtype=torch.float32).to(self.device)
                    _, _, action, _ = self.model(dummy_input, dummy_prev_output)
                    out.append(action[0, i].item())

                plt.plot(omega_vals.squeeze(), out, label=f'$u_{{{i+1}}}(\omega_{{{i+1}}})$', color=colors[i])

        plt.xlabel('$\\omega_i$ (Hz)', fontsize=12)
        plt.ylabel('$u_i(\\omega_i)$ (p.u.)', fontsize=12)
        # plt.xlim(-0.2, 0.2)
        plt.xlim(-0.025, 0.025)
        plt.title('Learned Monotonic Policies $u_i(\\omega_i)$', fontsize=14)
        plt.grid(True)
        plt.legend(ncol=2, fontsize=12)
        plt.tight_layout()
        # plt.savefig(figure_name, dpi=300)
        plt.savefig("Policy_Zoomed_in.png")
        print(f"[✓] Figure saved to {figure_name}")

def plot_u_from_weights(filepath, action_units=10):
    data = np.load(filepath)

    w_plus_temp0 = data['w_plus_temp0'] ** 2
    b_plus_temp0 = data['b_plus_temp0'] ** 2
    w_minus_temp0 = data['w_minus_temp0'] ** 2
    b_minus_temp0 = data['b_minus_temp0'] ** 2
    w_recover = data['w_recover']
    b_recover = data['b_recover']
    max_action = data['max_action']

    w_plus = np.matmul(w_plus_temp0, w_recover)
    b_plus = np.matmul(-b_plus_temp0, b_recover)
    w_minus = np.matmul(-w_minus_temp0, w_recover)
    b_minus = np.matmul(-b_minus_temp0, b_recover)

    omega_vals = np.linspace(-0.2, 0.2, 200)
    colors = plt.cm.tab10(np.linspace(0, 1, action_units))

    plt.figure(figsize=(10, 6))

    for i in range(action_units):
        u_vals = []
        for omega in omega_vals:
            x = omega * np.ones(w_plus.shape[1])

            relu_plus = np.maximum(x + b_plus[i], 0)
            relu_minus = np.maximum(-x + b_minus[i], 0)

            nonlinear_plus = np.sum(relu_plus * w_plus[i])
            nonlinear_minus = np.sum(relu_minus * w_minus[i])
            action_nonconstrained = nonlinear_plus + nonlinear_minus

            max_act = max_action[0, i] if max_action.ndim > 1 else max_action[i]
            u = max_act - max(0, max_act - action_nonconstrained) + max(0, -max_act - action_nonconstrained)
            u_vals.append(u)

        plt.plot(omega_vals, u_vals, label=f'$u_{{{i+1}}}(\omega_{{{i+1}}})$', color=colors[i])

    plt.xlabel('$\\omega_i$ (Hz)')
    plt.ylabel('$u_i(\\omega_i)$ (p.u.)')
    plt.title('Reconstructed Control Laws $u_i(\\omega_i)$ from Saved Weights')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("reconstructed_policies.png", dpi=300)
    print("[✓] Plot saved as 'reconstructed_policies.png'")

def plot_loss_record_from_file(filename="Training_Loss.png"):
    loss_record = np.load("relu_loss_record.npy")
    print(f"[✓] Loaded loss record from relu_loss_record.npy")

    plt.figure()
    plt.plot(loss_record, label='Training Loss')

    final_loss = loss_record[-1]
    plt.axhline(y=final_loss, linestyle='--', color='gray',\
                linewidth=1.2)
    plt.text(0.6*len(loss_record), final_loss+0.02, \
                f"{final_loss:.4f}", color='gray', fontsize=10)

    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.xlim(0, 1000)
    plt.title('Loss per Episode')
    plt.grid(True)
    plt.legend()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"[✓] Loss curve saved to {filename}")