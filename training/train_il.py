import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
import glob

# ─────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────

class DrivingDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.max_range = 10.0
        self.max_speed = 4.0

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        point = self.data[idx]

        lidar = np.array(point['lidar']) / self.max_range
        speed = np.array([point['speed'] / self.max_speed])
        state = np.concatenate([lidar, speed]).astype(np.float32)

        action = np.array([
            point['steering'],
            point['throttle']
        ], dtype=np.float32)

        return torch.tensor(state), torch.tensor(action)


# ─────────────────────────────────────────
# NEURAL NETWORK
# ─────────────────────────────────────────

class DrivingPolicy(nn.Module):
    def __init__(self, lidar_beams=36):
        super().__init__()
        input_dim = lidar_beams + 1  # LiDAR + speed

        self.network = nn.Sequential(
            nn.Linear(input_dim, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 2)   # steering + throttle
        )

    def forward(self, x):
        return self.network(x)


# ─────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────

def load_all_datasets(data_dir='/research_ws/data'):
    all_data = []
    pkl_files = glob.glob(os.path.join(data_dir, '*.pkl'))

    if not pkl_files:
        print(f'No .pkl files found in {data_dir}')
        return None

    for path in pkl_files:
        with open(path, 'rb') as f:
            data = pickle.load(f)
        all_data.extend(data)
        print(f'Loaded {len(data)} points from {path}')

    print(f'Total datapoints: {len(all_data)}')
    return all_data


# ─────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────

def train():
    # Load data
    raw_data = load_all_datasets()
    if raw_data is None:
        return

    # Print data statistics
    steerings = [d['steering'] for d in raw_data]
    throttles = [d['throttle'] for d in raw_data]
    print(f'Steering range: {min(steerings):.3f} to {max(steerings):.3f}')
    print(f'Throttle range: {min(throttles):.3f} to {max(throttles):.3f}')

    # Split 80% train, 20% validation
    split = int(0.8 * len(raw_data))
    np.random.shuffle(raw_data)
    train_data = raw_data[:split]
    val_data = raw_data[split:]

    train_loader = DataLoader(
        DrivingDataset(train_data), batch_size=64, shuffle=True
    )
    val_loader = DataLoader(
        DrivingDataset(val_data), batch_size=64, shuffle=False
    )

    print(f'Train: {len(train_data)} | Val: {len(val_data)}')

    # Build model
    model = DrivingPolicy(lidar_beams=36)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    best_val_loss = float('inf')
    os.makedirs('/research_ws/models', exist_ok=True)

    epochs = 150

    for epoch in range(epochs):

        # Training
        model.train()
        train_losses = []
        for states, actions in train_loader:
            predictions = model(states)
            loss = loss_fn(predictions, actions)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for states, actions in val_loader:
                predictions = model(states)
                loss = loss_fn(predictions, actions)
                val_losses.append(loss.item())

        avg_train = np.mean(train_losses)
        avg_val = np.mean(val_losses)

        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{epochs} | '
                  f'Train: {avg_train:.6f} | '
                  f'Val: {avg_val:.6f}')

        # Save best model
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(
                model.state_dict(),
                '/research_ws/models/il_policy.pth'
            )

    print(f'\nDone. Best val loss: {best_val_loss:.6f}')
    print('Model saved to /research_ws/models/il_policy.pth')


if __name__ == '__main__':
    train()
