import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from typing import Optional

class NextActivityModel(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int = 64, hidden: int = 128, pad_id: int = 0):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        self.lstm1 = nn.LSTM(emb_dim, hidden, batch_first=True)
        self.lstm2 = nn.LSTM(hidden, hidden // 2, batch_first=True)
        self.fc1 = nn.Linear(hidden // 2, 128)
        self.fc2 = nn.Linear(128, 64)
        self.out = nn.Linear(64, vocab_size)

    def forward(self, x):
        e = self.emb(x)               # (B, T, E)
        o1, _ = self.lstm1(e)         # (B, T, H)
        o2, _ = self.lstm2(o1)        # (B, T, H/2)
        last = o2[:, -1, :]           # (B, H/2)
        z = F.relu(self.fc1(last))
        z = F.relu(self.fc2(z))
        return self.out(z)            # (B, vocab)


class RemainingTimeModel(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int = 32, hidden: int = 96, pad_id: int = 0):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        self.lstm1 = nn.LSTM(emb_dim, hidden, batch_first=True)
        self.lstm2 = nn.LSTM(hidden, hidden // 2, batch_first=True)
        self.fc1 = nn.Linear(hidden // 2, 64)
        self.fc2 = nn.Linear(64, 32)
        self.out = nn.Linear(32, 1)

    def forward(self, x):
        e = self.emb(x)
        o1, _ = self.lstm1(e)
        o2, _ = self.lstm2(o1)
        last = o2[:, -1, :]
        z = F.relu(self.fc1(last))
        z = F.relu(self.fc2(z))
        return self.out(z).squeeze(-1)  # (B,)


class TorchCombinedPredictor:
    def __init__(self, vocab_size: int, pad_id: int = 0, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.next_model = NextActivityModel(vocab_size, pad_id=pad_id).to(self.device)
        self.time_model = RemainingTimeModel(vocab_size, pad_id=pad_id).to(self.device)

        # AMP = سرعت بیشتر روی GPU
        self.use_amp = (self.device == "cuda")
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

    def train_next(self, loader, epochs: int = 5, lr: float = 1e-3):
        opt = torch.optim.Adam(self.next_model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        self.next_model.train()

        for ep in range(1, epochs + 1):
            t0 = time.perf_counter()
            total_loss, total, correct = 0.0, 0, 0
            for x, y_next, _ in loader:
                x = x.to(self.device, non_blocking=True)
                y_next = y_next.to(self.device, non_blocking=True)

                opt.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    logits = self.next_model(x)
                    loss = criterion(logits, y_next)

                self.scaler.scale(loss).backward()
                self.scaler.step(opt)
                self.scaler.update()

                total_loss += loss.item() * x.size(0)
                pred = logits.argmax(dim=1)
                correct += (pred == y_next).sum().item()
                total += x.size(0)

            dt = time.perf_counter() - t0
            print(f"[Next] Epoch {ep}/{epochs} | loss={total_loss/total:.4f} | acc={correct/total:.4f} | epoch_time={dt:.1f}s")

    def train_time(self, loader, epochs: int = 5, lr: float = 1e-3):
        opt = torch.optim.Adam(self.time_model.parameters(), lr=lr)
        criterion = nn.L1Loss()  # MAE
        self.time_model.train()

        for ep in range(1, epochs + 1):
            t0 = time.perf_counter()
            total_loss, total = 0.0, 0
            for x, _, y_time in loader:
                x = x.to(self.device, non_blocking=True)
                y_time = y_time.to(self.device, non_blocking=True)

                opt.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    pred = self.time_model(x)
                    loss = criterion(pred, y_time)

                self.scaler.scale(loss).backward()
                self.scaler.step(opt)
                self.scaler.update()

                total_loss += loss.item() * x.size(0)
                total += x.size(0)
            
            dt = time.perf_counter() - t0
            print(f"[Time] Epoch {ep}/{epochs} | mae={total_loss/total:.4f} | epoch_time={dt:.1f}s")

    def save(self, out_dir: str):
        os.makedirs(out_dir, exist_ok=True)
        torch.save(self.next_model.state_dict(), os.path.join(out_dir, "next_activity.pt"))
        torch.save(self.time_model.state_dict(), os.path.join(out_dir, "remaining_time.pt"))

    def load(self, model_dir: str):
        next_path = os.path.join(model_dir, "next_activity.pt")
        time_path = os.path.join(model_dir, "remaining_time.pt")
        if not os.path.exists(next_path) or not os.path.exists(time_path):
            raise FileNotFoundError("Torch model weights not found")

        self.next_model.load_state_dict(torch.load(next_path, map_location=self.device))
        self.time_model.load_state_dict(torch.load(time_path, map_location=self.device))
        self.next_model.eval()
        self.time_model.eval()

    def _to_tensor(self, seq):
        if isinstance(seq, torch.Tensor):
            return seq.to(self.device)
        return torch.as_tensor(seq, dtype=torch.long, device=self.device)

    @torch.no_grad()
    def predict_next(self, seq, return_proba: bool = False):
        x = self._to_tensor(seq)
        if x.ndim == 1:
            x = x.unsqueeze(0)
        logits = self.next_model(x)
        if return_proba:
            return torch.softmax(logits, dim=-1).cpu().numpy()
        return logits.argmax(dim=-1).cpu().numpy()

    @torch.no_grad()
    def predict_remaining_time(self, seq):
        x = self._to_tensor(seq)
        if x.ndim == 1:
            x = x.unsqueeze(0)
        pred = self.time_model(x)
        return pred.cpu().numpy()
