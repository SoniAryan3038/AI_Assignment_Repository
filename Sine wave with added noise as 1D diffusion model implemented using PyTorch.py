import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Generate a noisy sine wave signal
def generate_noisy_sine_wave(length, noise_level=0.1):
    x = np.linspace(0, 4 * np.pi, length) #creating a numpy array of desired values
    y = np.sin(x) #Function of sin(x)
    noise = noise_level * np.random.randn(length)
    return y + noise

# Diffusion Model Class
class DiffusionModel(nn.Module):
    def __init__(self, diffusion_steps=50):
        super(DiffusionModel, self).__init__()
        self.diffusion_steps = diffusion_steps
        self.linear1 = nn.Linear(1, 64)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(64, 1)

    def forward(self, x, t):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

    def diffusion_process(self, signal):
        noisy_signals = [signal]
        for t in range(self.diffusion_steps):
            noise = 0.1 * torch.randn_like(signal)
            signal = signal + noise
            noisy_signals.append(signal)
        return noisy_signals

    def reverse_diffusion(self, noisy_signal):
        for t in reversed(range(self.diffusion_steps)):
            noisy_signal = self.forward(noisy_signal, t)
        return noisy_signal

# Generate data
length = 100
original_signal = generate_noisy_sine_wave(length)
original_signal_tensor = torch.tensor(original_signal, dtype=torch.float32).unsqueeze(-1)

# Initialize model
diffusion_model = DiffusionModel(diffusion_steps=50)

# Apply diffusion process
noisy_signals = diffusion_model.diffusion_process(original_signal_tensor)

# Reverse diffusion process
reconstructed_signal = diffusion_model.reverse_diffusion(noisy_signals[-1])

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(original_signal, label='Original Signal')
plt.plot(noisy_signals[-1].detach().numpy(), label='Noisy Signal')
plt.plot(reconstructed_signal.detach().numpy(), label='Reconstructed Signal')
plt.legend()
plt.show()
