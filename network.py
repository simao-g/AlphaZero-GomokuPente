from typing import Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = F.relu(out)
        return out


class AlphaZeroNet(nn.Module):
    """
    Small AlphaZero-style residual network.

    Params:
    - in_channels: input channels (3 by default)
    - board_size: H == W == board_size
    - action_size: number of discrete actions (board_size * board_size)
    - n_res_blocks: number of residual blocks (tune to trade speed vs performance)
    - channels: number of convolutional filters in body
    """

    def __init__(self,
                 in_channels: int = 3,
                 board_size: int = 15,
                 action_size: int = 15*15,
                 n_res_blocks: int = 6,
                 channels: int = 128):
        super().__init__()

        self.board_size = board_size
        self.action_size = action_size
        self.channels = channels

        # initial conv block
        self.conv = nn.Conv2d(in_channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(channels)

        # residual tower
        self.res_blocks = nn.ModuleList([ResidualBlock(channels) for _ in range(n_res_blocks)])

        # policy head
        # small conv -> flatten -> linear to action_size
        self.policy_conv = nn.Conv2d(channels, 2, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * board_size * board_size, action_size)

        # value head
        # small conv -> flatten -> two-layer MLP -> scalar
        self.value_conv = nn.Conv2d(channels, 1, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(board_size * board_size, 64)
        self.value_fc2 = nn.Linear(64, 1)

        self._init_weights()

    def _init_weights(self):
        # Kaiming init for convs
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Input:
            x: (B, in_channels, H, W) float32
        Returns:
            policy_logits: (B, action_size)  (raw logits, before softmax)
            value: (B, 1) float in [-1,1] after tanh
        """
        # body
        out = self.conv(x)
        out = self.bn(out)
        out = F.relu(out)

        for blk in self.res_blocks:
            out = blk(out)

        # policy head
        p = self.policy_conv(out)
        p = self.policy_bn(p)
        p = F.relu(p)
        p = p.view(p.shape[0], -1)
        policy_logits = self.policy_fc(p)  # (B, action_size)

        # value head
        v = self.value_conv(out)
        v = self.value_bn(v)
        v = F.relu(v)
        v = v.view(v.shape[0], -1)
        v = F.relu(self.value_fc1(v))
        v = self.value_fc2(v)
        value = torch.tanh(v)

        return policy_logits, value
    
    def predict(self, state):
        """
        Método de previsão que recebe o estado e retorna a política e o valor.
        """
        # Converte o estado (se ainda for um numpy array) para tensor do PyTorch
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # Adiciona dimensão extra para batch size (1)

        # Passa o tensor pela rede
        policy_logits, value = self(state_tensor)

        return policy_logits, value


class PyTorchModel:
    """
    Wrapper around AlphaZeroNet that provides:
     - predict(encoded_states) -> (policy_np, value_np)
     - predict_batch(list_of_encoded_states) -> (policy_np, value_np)
     - train_batch(...) -> loss dict and backprop step
     - save / load
    """

    def __init__(self,
                 board_size: int = 15,
                 action_size: Optional[int] = None,
                 device: Optional[str] = None,
                 n_res_blocks: int = 3,
                 channels: int = 64,
                 lr: float = 1e-3,
                 weight_decay: float = 1e-4):
        self.board_size = board_size
        self.action_size = action_size if action_size is not None else board_size * board_size

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.net = AlphaZeroNet(
            in_channels=3,
            board_size=board_size,
            action_size=self.action_size,
            n_res_blocks=n_res_blocks,
            channels=channels
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr, weight_decay=weight_decay)
        self.value_loss_fn = nn.MSELoss()
        self.policy_loss_fn = nn.KLDivLoss(reduction='batchmean')  # log_probs vs target probs

    # -------------------------
    # Prediction (batch for MCTS)
    # -------------------------
    def predict(self, encoded_states: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        encoded_states: numpy array (B, C, H, W) float32
        returns:
            policy_probs: np.array (B, action_size) float32
            values: np.array (B, 1) float32
        """
        training = self.net.training
        self.net.eval()
        with torch.no_grad():
            x = torch.from_numpy(encoded_states.astype(np.float32)).to(self.device)
            logits, value = self.net(x)
            probs = F.softmax(logits, dim=1).cpu().numpy()
            values = value.cpu().numpy()
        self.net.train(training)
        return probs, values

    # -------------------------
    # Convenience: batch from list
    # -------------------------
    def predict_batch(self, states_list: list) -> Tuple[np.ndarray, np.ndarray]:
        """
        states_list: list of np arrays (C,H,W)
        returns: policy_probs, values as np arrays
        """
        batch = self.make_batch_from_states(states_list)
        return self.predict(batch)

    # -------------------------
    # Single train batch
    # -------------------------
    def train_batch(self,
                    states: np.ndarray,
                    target_pis: np.ndarray,
                    target_vs: np.ndarray,
                    epochs: int = 1) -> dict:
        self.net.train()
        states_t = torch.from_numpy(states.astype(np.float32)).to(self.device)
        target_pis_t = torch.from_numpy(target_pis.astype(np.float32)).to(self.device)
        target_vs_t = torch.from_numpy(target_vs.astype(np.float32)).to(self.device)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_loss = 0.0

        for _ in range(epochs):
            self.optimizer.zero_grad()
            logits, values = self.net(states_t)
            log_probs = F.log_softmax(logits, dim=1)

            policy_loss = self.policy_loss_fn(log_probs, target_pis_t)
            value_loss = self.value_loss_fn(values, target_vs_t)

            loss = policy_loss + value_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), 3.0)
            self.optimizer.step()

            total_policy_loss += float(policy_loss.item())
            total_value_loss += float(value_loss.item())
            total_loss += float(loss.item())

        ne = float(epochs)
        return {
            "policy_loss": total_policy_loss / ne,
            "value_loss": total_value_loss / ne,
            "total_loss": total_loss / ne
        }

    # -------------------------
    # Save / Load
    # -------------------------
    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        state = {
            "net": self.net.state_dict(),
            "opt": self.optimizer.state_dict(),
            "board_size": self.board_size,
            "action_size": self.action_size
        }
        torch.save(state, path)

    def load(self, path: str, map_location: Optional[str] = None) -> None:
        map_location = map_location or self.device
        state = torch.load(path, map_location=map_location)
        self.net.load_state_dict(state["net"])
        if "opt" in state and state["opt"] is not None:
            try:
                self.optimizer.load_state_dict(state["opt"])
            except Exception:
                pass

    # -------------------------
    # Utility: convert list -> batch
    # -------------------------
    @staticmethod
    def make_batch_from_states(list_of_encoded_states: list) -> np.ndarray:
        return np.stack(list_of_encoded_states, axis=0).astype(np.float32)