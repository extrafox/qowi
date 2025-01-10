import torch
import torch.nn as nn
import torch.optim as optim
import random

def haar_loss(encoded_output, reconstructed_output, input_data, target_haar_coeffs):
    """
    Loss function with constraints:
    1. Minimize difference between forward 2D Haar coefficients and encoded nn_haar codes.
    2. Minimize difference between reconstructed pixels and input spatial pixels.
    """
    # Forward Haar loss: Compare network's output with target Haar coefficients
    forward_loss = nn.MSELoss()(encoded_output, target_haar_coeffs)

    # Reconstruction loss: Ensure the reconstructed input matches the original input
    recon_loss = nn.MSELoss()(reconstructed_output, input_data)

    # Weighted combined loss
    total_loss = 0.3 * forward_loss + 0.7 * recon_loss
    return total_loss

class SimpleHaarNet(nn.Module):
    def __init__(self):
        super(SimpleHaarNet, self).__init__()
        self.fc1 = nn.Linear(4, 16)  # Input size is 4 spatial values
        self.bn1 = nn.BatchNorm1d(16)
        self.fc2 = nn.Linear(16, 16)
        self.bn2 = nn.BatchNorm1d(16)
        self.fc3 = nn.Linear(16, 4)  # Output size is 4 Haar coefficients

    def forward(self, x):
        if x.dim() == 2 and x.size(0) == 1:  # Single sample
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
        else:
            x = torch.relu(self.bn1(self.fc1(x)))
            x = torch.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        return x

def compute_target_haar_coeffs(input_data):
    """
    Compute the 2D Haar transform coefficients (target values) from spatial input pixels.
    """
    target_coeffs = []
    for sample in input_data:
        a, b, c, d = sample
        LL = (a + b + c + d) / 2
        HL = (a + b - c - d) / 2
        LH = (a - b + c - d) / 2
        HH = (a - b - c + d) / 2
        target_coeffs.append([LL, HL, LH, HH])
    return torch.tensor(target_coeffs, dtype=torch.float32)

# Training script

def train_network():
    # Define the networks
    encode_net = SimpleHaarNet()  # Neural network for forward Haar encoding
    decode_net = SimpleHaarNet()  # Neural network for inverse Haar decoding

    # Optimizers for both networks
    optimizer_encode = optim.Adam(encode_net.parameters(), lr=0.0005)
    optimizer_decode = optim.Adam(decode_net.parameters(), lr=0.0005)

    # Learning rate schedulers
    scheduler_encode = optim.lr_scheduler.ReduceLROnPlateau(optimizer_encode, patience=10)
    scheduler_decode = optim.lr_scheduler.ReduceLROnPlateau(optimizer_decode, patience=10)

    for epoch in range(1000):  # Number of training epochs
        # Generate a new random dataset for each epoch
        input_data = torch.tensor([
            [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]
            for _ in range(1000)
        ], dtype=torch.float32)  # Batch of spatial domain values

        # Compute target Haar coefficients
        target_haar_coeffs = compute_target_haar_coeffs(input_data)

        # Forward pass through the encode network
        optimizer_encode.zero_grad()
        encoded_output = encode_net(input_data)

        # Forward pass through the decode network
        reconstructed_output = decode_net(encoded_output)

        # Compute loss
        loss = haar_loss(encoded_output, reconstructed_output, input_data, target_haar_coeffs)

        # Backpropagation and optimization for both networks
        loss.backward()
        optimizer_encode.step()
        optimizer_decode.step()

        # Step the schedulers
        scheduler_encode.step(loss)
        scheduler_decode.step(loss)

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")

    return encode_net, decode_net

# Example usage
encode_net, decode_net = train_network()
input_data = [100.0, 120.0, 130.0, 110.0]
input_sample = torch.tensor([input_data])
target_coeffs = compute_target_haar_coeffs(input_sample)
encoded = encode_net(input_sample)
reconstructed = decode_net(encoded)
reconstructed = torch.clamp(reconstructed, 0, 255)  # Clamp reconstructed values to valid pixel range

print("Sample Input (Spatial Values):", input_data)
print("Target Haar Coefficients:", target_coeffs)
print("Encoded (Haar Coefficients):", encoded)
print("Reconstructed (Spatial Values):", reconstructed)
