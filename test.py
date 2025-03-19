import torch
import torch.nn as nn
import torch.optim as optim
import random

# -------------------------
# 1. Define an energy model
# -------------------------
class EnergyModel(nn.Module):
    """
    Example Energy Model (E_\theta) that outputs a scalar energy for each sample.
    Modify the architecture for your specific data (e.g., 3D CNN for videos).
    """
    def __init__(self):
        super(EnergyModel, self).__init__()
        
        # Example: a small feedforward network for demonstration
        # Replace this with something suitable for video data
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        """
        x: (batch_size, feature_dimension) or frames stacked
        returns: energies of shape (batch_size,)
        """
        return self.fc(x).squeeze(-1)  # scalar output per sample

# -----------------------------
# 2. Langevin dynamics function
# -----------------------------
def sample_langevin(model, x_init, K, step_size, sigma):
    """
    Runs K steps of Langevin dynamics to produce negative samples.
    Args:
        model      : EnergyModel
        x_init     : Initial samples (requires grad)
        K          : Number of Langevin steps
        step_size  : Step size (learning rate in pixel/feature space)
        sigma      : Std of Gaussian noise at each step
    Returns:
        x_neg      : Final samples (negative samples)
    """
    x_neg = x_init.clone().detach()
    x_neg.requires_grad_(True)

    for k in range(K):
        # Forward pass: compute energy
        energy = model(x_neg)
        # Compute gradient of the energy wrt x_neg
        grad = torch.autograd.grad(energy.sum(), x_neg, retain_graph=False)[0]
        # Gradient ascent on log probability corresponds to gradient descent on energy
        x_neg = x_neg - step_size * grad + sigma * torch.randn_like(x_neg)
        # If necessary, clamp or project x_neg back into valid range. 
        # For demonstration, assume inputs range in [-1,1], for instance:
        x_neg = torch.clamp(x_neg, -1, 1)
        x_neg.requires_grad_(True)

    # Detach the final sample
    x_neg = x_neg.detach()
    return x_neg

# --------------------------------
# 3. Training loop and replay buffer
# --------------------------------
def train_energy_model(
        model, 
        data_loader,         # An iterator over real data samples x^+
        buffer_size=1000,    # Max size of replay buffer
        replay_prob=0.95,    # Probability of sampling from replay buffer
        K=10,                # Number of Langevin steps
        step_size=0.1,       # Langevin step size
        sigma=0.01,          # Langevin noise scale
        alpha=0.1,           # Weight for the L2 term in the loss
        lr=1e-4,             # Learning rate for Adam
        num_epochs=10        # Number of training epochs
    ):
    """
    Main training loop for the energy-based model with replay buffer.
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Initialize the replay buffer (store as a Python list)
    replay_buffer = []
    
    for epoch in range(num_epochs):
        for real_batch in data_loader:
            # -----------------------------------------------------
            # (a) Sample positive examples x_i^{+} from real data
            # -----------------------------------------------------
            x_pos = real_batch.to(next(model.parameters()).device)
            
            # ---------------------------------------------------------
            # (b) Initialize negative samples x_i^0 from buffer or noise
            # ---------------------------------------------------------
            current_batch_size = x_pos.shape[0]
            x_init_list = []
            
            for _ in range(current_batch_size):
                # With 95% probability, sample from replay buffer
                if len(replay_buffer) > 0 and random.random() < replay_prob:
                    x_init_list.append(random.choice(replay_buffer).to(x_pos.device))
                else:
                    # Otherwise sample from uniform or any other distribution
                    # For demonstration, uniform in [-1,1] for same shape
                    x_init_list.append(torch.rand_like(x_pos[0], device=x_pos.device) * 2 - 1)
            
            x_init = torch.stack(x_init_list).to(x_pos.device)
            
            # -----------------------------
            # (c) Langevin dynamics
            # -----------------------------
            x_neg = sample_langevin(model, x_init, K, step_size, sigma)
            
            # Optionally apply some post-processing: \Omega(\cdot)
            # In many cases, we might clamp or transform. We assume clamp is already done inside sample_langevin.
            
            # -----------------------------------
            # (d) Compute the loss and gradients
            # -----------------------------------
            
            # E_\theta(x^+)
            e_pos = model(x_pos)
            # E_\theta(x^-)
            e_neg = model(x_neg)
            
            # L2 part: E_\theta(x^+)^2 + E_\theta(x^-)^2
            loss_l2 = (e_pos**2 + e_neg**2).mean()
            # ML part: E_\theta(x^+) - E_\theta(x^-)
            loss_ml = (e_pos - e_neg).mean()
            
            loss = alpha * loss_l2 + loss_ml
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # ---------------------------------------
            # (e) Add negative samples to replay buffer
            # ---------------------------------------
            # Keep memory usage in check by limiting buffer size
            x_neg_list = list(x_neg.detach().cpu())
            replay_buffer.extend(x_neg_list)
            if len(replay_buffer) > buffer_size:
                replay_buffer = replay_buffer[-buffer_size:]  # Keep recent samples only
            
        print(f"Epoch [{epoch+1}/{num_epochs}] | Loss: {loss.item():.4f}")


if __name__ == "__main__":
    # Create a dummy dataset loader
    # Suppose each batch is (batch_size, 128) in [-1, 1]
    # In practice, this would be your video data (e.g., frames or features).
    batch_size = 16
    num_samples = 1000
    dummy_data = (torch.rand(num_samples, 128) * 2 - 1)  # in [-1,1]
    data_loader = torch.utils.data.DataLoader(dummy_data, batch_size=batch_size, shuffle=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Instantiate the model
    model = EnergyModel().to(device)
    
    # Train the model
    train_energy_model(
        model,
        data_loader,
        buffer_size=500,
        replay_prob=0.95,
        K=10,
        step_size=0.05,
        sigma=0.01,
        alpha=0.1,
        lr=1e-4,
        num_epochs=2000
    )
