import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
import random
import gym
from torch.utils.data import DataLoader, Dataset

# For wandb logging
from pytorch_lightning.loggers import WandbLogger

# Fix potential numpy attribute issue if needed.
if not hasattr(np, "bool8"):
    np.bool8 = np.bool_


# --------------------
# Replay Buffer
# --------------------
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []

    def add(self, obs, action, reward, done, next_obs):
        self.buffer.append((obs, action, reward, done, next_obs))
        if len(self.buffer) > self.capacity:
            self.buffer.pop(0)

    def __len__(self):
        return len(self.buffer)

    def sample_sequence(self, seq_length):
        if len(self.buffer) < seq_length:
            return None
        start = random.randint(0, len(self.buffer) - seq_length)
        return self.buffer[start : start + seq_length]


# --------------------
# Diffusion Model Module
# --------------------
class DiffusionModel(nn.Module):
    def __init__(self, obs_dim, hidden_dim):
        super().__init__()
        # A simple MLP that takes in a noisy observation and a noise-level (tau)
        # and outputs the predicted “denoised” observation.
        self.net = nn.Sequential(
            nn.Linear(obs_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, obs_dim),
        )

    def forward(self, x_noisy, tau):
        # x_noisy: (batch, obs_dim); tau: (batch, 1)
        inp = torch.cat([x_noisy, tau], dim=-1)
        return self.net(inp)

    def reverse_process(self, x, tau):
        # Dummy reverse diffusion: here we simply call forward.
        # In practice, this should simulate the reverse diffusion process.
        return self.forward(x, tau)


# --------------------
# Reward/Termination Predictor
# --------------------
class RewardEndModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_reward_classes=3):
        """
        We assume reward prediction is a classification task (e.g. negative, zero, positive)
        and termination is binary.
        """
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.reward_head = nn.Linear(hidden_dim, num_reward_classes)
        self.done_head = nn.Linear(hidden_dim, 2)

    def forward(self, x_seq, hidden=None):
        # x_seq: (batch, seq_len, input_dim)
        out, hidden = self.lstm(x_seq, hidden)
        reward_logits = self.reward_head(out)  # (batch, seq_len, num_reward_classes)
        done_logits = self.done_head(out)  # (batch, seq_len, 2)
        return reward_logits, done_logits, hidden


# --------------------
# Actor and Critic Modules
# --------------------
class ActorNetwork(nn.Module):
    def __init__(self, obs_dim, hidden_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, obs):
        logits = self.net(obs)
        return logits

    def sample_action(self, obs):
        logits = self.forward(obs)
        probs = F.softmax(logits, dim=-1)
        action = torch.multinomial(probs, num_samples=1)
        return action.item()


class CriticNetwork(nn.Module):
    def __init__(self, obs_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )

    def forward(self, obs):
        value = self.net(obs)
        return value


# --------------------
# Dummy Dataset (since we use our own experience collection)
# --------------------
class DummyDataset(Dataset):
    def __init__(self, length=1000):
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return torch.tensor(0)  # dummy


# --------------------
# Main Lightning Module with WandB Logging
# --------------------
class RLLightningModule(pl.LightningModule):
    def __init__(
        self,
        env_name="CartPole-v1",
        buffer_capacity=10000,
        steps_collect=100,
        steps_diffusion_model=1000,
        steps_reward_end_model=1000,
        steps_actor_critic=1000,
        L=4,
        H=4,
        lr_diffusion=1e-3,
        lr_reward=1e-3,
        lr_actor=1e-3,
        P_mean=0.0,
        P_std=1.0,
        hidden_dim=64,
    ):
        """
        L: length of history used in diffusion and reward/end updates.
        H: horizon length for actor–critic imagination.
        P_mean, P_std: parameters for the log-normal sigma (as in EDM).
        """
        super().__init__()
        self.save_hyperparameters()
        self.env = gym.make(env_name)
        obs_space = self.env.observation_space
        act_space = self.env.action_space
        self.obs_dim = obs_space.shape[0]
        self.action_dim = act_space.n

        self.buffer = ReplayBuffer(buffer_capacity)

        # Initialize our sub-modules.
        self.diffusion_model = DiffusionModel(self.obs_dim, hidden_dim)
        # For the reward/end module, we use an input combining observation and one-hot action.
        self.reward_end_model = RewardEndModel(
            self.obs_dim + self.action_dim, hidden_dim
        )
        self.actor = ActorNetwork(self.obs_dim, hidden_dim, self.action_dim)
        self.critic = CriticNetwork(self.obs_dim, hidden_dim)

        # We'll perform manual optimization.
        self.automatic_optimization = False

    def train_dataloader(self):
        # We use a dummy dataloader because our training_step does its own experience collection.
        return DataLoader(DummyDataset(), batch_size=1)

    # --------------------
    # Experience Collection
    # --------------------
    def collect_experience(self, n_steps):
        obs, info = self.env.reset()
        for t in range(n_steps):
            # Convert to tensor (with batch dimension)
            obs_tensor = torch.tensor(
                obs, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            action = self.actor.sample_action(obs_tensor)
            next_obs, reward, done, _, _ = self.env.step(action)
            self.buffer.add(obs, action, reward, done, next_obs)
            if done:
                obs, info = self.env.reset()
            else:
                obs = next_obs

    # --------------------
    # Diffusion Model Update
    # --------------------
    def update_diffusion_model(self):
        # We sample a contiguous sequence of length L+2:
        # (x_{t-L+1}, a_{t-L+1}, …, x_t, a_t, x_{t+1})
        seq_length = self.hparams.L + 2
        seq = self.buffer.sample_sequence(seq_length)
        if seq is None:
            return torch.tensor(0.0, device=self.device)
        # For simplicity, we only use the last transition: x_{t+1} is our target.
        target = torch.tensor(
            seq[-1][4], dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        # Sample a noise level sigma using a log-normal parameterization.
        log_sigma = (
            torch.randn(1, device=self.device) * self.hparams.P_std
            + self.hparams.P_mean
        )
        sigma = log_sigma.exp()
        tau = sigma  # identity schedule (tau = sigma)
        # Create a noisy version of the target.
        noise = torch.randn_like(target) * sigma
        x_noisy = target + noise
        tau_tensor = tau.unsqueeze(0)  # shape (1, 1)
        # Predict the original observation.
        pred = self.diffusion_model(x_noisy, tau_tensor)
        loss = F.mse_loss(pred, target)

        # Manual optimizer step for diffusion model (optimizer index 0).
        optimizer = self.optimizers()[0]
        optimizer.zero_grad()
        self.manual_backward(loss)
        optimizer.step()
        return loss.detach()

    # --------------------
    # Reward/Termination Model Update
    # --------------------
    def update_reward_end_model(self):
        # Sample a sequence of length L+H (burn-in + horizon)
        seq_length = self.hparams.L + self.hparams.H
        seq = self.buffer.sample_sequence(seq_length)
        if seq is None:
            return torch.tensor(0.0, device=self.device)

        inputs, reward_targets, done_targets = [], [], []
        for obs, action, reward, done, next_obs in seq:
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device)
            action_onehot = torch.zeros(self.action_dim, device=self.device)
            action_onehot[action] = 1.0
            inp = torch.cat([obs_tensor, action_onehot], dim=-1)
            inputs.append(inp.unsqueeze(0))
            # Map reward sign to a class: for example, -1 -> 0, 0 -> 1, 1 -> 2.
            sign_reward = np.sign(reward)
            r_class = 0 if sign_reward < 0 else 1 if sign_reward == 0 else 2
            reward_targets.append(r_class)
            done_targets.append(int(done))

        inputs = torch.cat(inputs, dim=0).unsqueeze(0)  # (1, seq_length, input_dim)
        reward_targets = torch.tensor(
            reward_targets, dtype=torch.long, device=self.device
        ).unsqueeze(0)
        done_targets = torch.tensor(
            done_targets, dtype=torch.long, device=self.device
        ).unsqueeze(0)

        reward_logits, done_logits, _ = self.reward_end_model(inputs)
        loss_reward = F.cross_entropy(
            reward_logits.view(-1, reward_logits.size(-1)), reward_targets.view(-1)
        )
        loss_done = F.cross_entropy(
            done_logits.view(-1, done_logits.size(-1)), done_targets.view(-1)
        )
        loss = loss_reward + loss_done

        # Manual optimizer step for reward/end model (optimizer index 1).
        optimizer = self.optimizers()[1]
        optimizer.zero_grad()
        self.manual_backward(loss)
        optimizer.step()
        return loss.detach()

    # --------------------
    # Actor–Critic Update (with imagination)
    # --------------------
    def update_actor_critic(self):
        # Sample an initial burn-in sequence of length L.
        seq = self.buffer.sample_sequence(self.hparams.L)
        if seq is None:
            return torch.tensor(0.0, device=self.device)
        # Use the last transition’s next observation as the starting point.
        current_obs = torch.tensor(
            seq[-1][4], dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        total_actor_loss = 0.0
        total_critic_loss = 0.0
        gamma = 0.99
        values, log_probs, rewards = [], [], []

        # Simulate H steps using the current policy and our learned models.
        for i in range(self.hparams.H):
            # Actor: sample an action.
            logits = self.actor(current_obs)
            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            log_probs.append(log_prob)

            # Use the reward/end model to “imagine” a reward and termination.
            # Create an input (current_obs + one-hot action) for the LSTM.
            action_onehot = torch.zeros(1, self.action_dim, device=self.device)
            action_onehot[0, action] = 1.0
            r_inp = torch.cat([current_obs, action_onehot], dim=-1).unsqueeze(
                0
            )  # (1,1,input_dim)
            reward_logits, done_logits, _ = self.reward_end_model(r_inp)
            reward_probs = F.softmax(reward_logits.squeeze(0), dim=-1)
            # Map classes to reward values (e.g. [-1, 0, 1])
            class_vals = torch.tensor([-1.0, 0.0, 1.0], device=self.device)
            predicted_reward = (reward_probs * class_vals).sum()
            rewards.append(predicted_reward)

            # Critic: get value estimate.
            value = self.critic(current_obs)
            values.append(value)

            # Simulate the next observation via a (dummy) reverse diffusion step.
            tau_dummy = torch.tensor([[0.1]], device=self.device)
            current_obs = self.diffusion_model.reverse_process(current_obs, tau_dummy)

            # (Optionally break if termination probability is high.)
            done_prob = F.softmax(done_logits.squeeze(0).squeeze(0), dim=-1)[1]
            if done_prob.item() > 0.5:
                break

        # Compute returns (backwards discounted sum).
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)
        returns = torch.stack(returns)
        values = torch.cat(values).squeeze(-1)
        log_probs = torch.stack(log_probs)
        advantage = returns - values.detach()

        actor_loss = -(log_probs * advantage).mean()
        critic_loss = F.mse_loss(values, returns)
        loss = actor_loss + critic_loss

        # Manual optimizer step for actor–critic (optimizer index 2).
        optimizer = self.optimizers()[2]
        optimizer.zero_grad()
        self.manual_backward(loss)
        optimizer.step()
        return loss.detach()

    # --------------------
    # Training Step (orchestrates the whole loop)
    # --------------------
    def training_step(self, batch, batch_idx):
        # 1. Collect experience (e.g. for a fixed number of environment steps)
        self.collect_experience(self.hparams.steps_collect)

        # 2. Update the diffusion model (e.g. several gradient steps)
        loss_diff = 0.0
        for _ in range(self.hparams.steps_diffusion_model):
            loss_diff = self.update_diffusion_model()
            self.log("loss_diffusion", loss_diff, on_step=True, prog_bar=True)

        # 3. Update the reward/termination model.
        loss_reward = 0.0
        for _ in range(self.hparams.steps_reward_end_model):
            loss_reward = self.update_reward_end_model()
            self.log("loss_reward_end", loss_reward, on_step=True, prog_bar=True)

        # 4. Update the actor–critic.
        loss_actor = 0.0
        for _ in range(self.hparams.steps_actor_critic):
            loss_actor = self.update_actor_critic()
            self.log("loss_actor_critic", loss_actor, on_step=True, prog_bar=True)

        total_loss = loss_diff + loss_reward + loss_actor
        self.log("total_loss", total_loss, on_step=True, prog_bar=True)
        return {"loss": total_loss}

    # --------------------
    # Optimizers for each module
    # --------------------
    def configure_optimizers(self):
        optimizer_diffusion = torch.optim.Adam(
            self.diffusion_model.parameters(), lr=self.hparams.lr_diffusion
        )
        optimizer_reward = torch.optim.Adam(
            self.reward_end_model.parameters(), lr=self.hparams.lr_reward
        )
        optimizer_actor = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=self.hparams.lr_actor,
        )
        return [optimizer_diffusion, optimizer_reward, optimizer_actor]


# --------------------
# Main execution with WandB Logger
# --------------------
if __name__ == "__main__":
    # Initialize WandB Logger.
    wandb_logger = WandbLogger(project="my_rl_project")

    model = RLLightningModule()

    # Pass the logger to the Trainer.
    trainer = pl.Trainer(max_epochs=10, logger=wandb_logger, devices=1)
    trainer.fit(model)
