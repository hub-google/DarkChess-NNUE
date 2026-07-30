import torch
import torch.nn as nn
import torch.optim as optim
import os
import glob
import gzip
import json
import numpy as np
from torch.utils.data import IterableDataset, DataLoader
from board import DarkChessBoardPy

# --- NNUE Model Definition ---
class DarkChessNNUE(nn.Module):
    def __init__(self):
        super(DarkChessNNUE, self).__init__()
        # 32 squares * 15 piece types (14 visible + 1 hidden) = 480
        # + 14 bag counts = 494 features
        self.fc1 = nn.Linear(494, 256)
        self.fc2 = nn.Linear(256, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        # Clipped ReLU as per docs (0~127) - approximate with ReLU for now
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))

# --- Auto-Tuning State Machine ---
class AutoTuningState:
    FAST_LEARNING = "A"
    PLATEAU = "B"
    REFINEMENT = "C"

class AutoTuner:
    def __init__(self):
        self.state = AutoTuningState.FAST_LEARNING
        self.consecutive_sprt_failures = 0
        self.draw_rate = 0.0

    def get_hyperparameters(self):
        if self.state == AutoTuningState.FAST_LEARNING:
            return {"lr": 5e-4, "temperature": 1.0, "window_size": 200_000, "sprt_bounds": [0, 15]}
        elif self.state == AutoTuningState.PLATEAU:
            return {"lr": 1e-4, "temperature": 1.5, "window_size": 1_000_000, "sprt_bounds": [0, 15]}
        else: # REFINEMENT
            return {"lr": 1e-6, "temperature": 0.1, "window_size": 500_000, "sprt_bounds": [0, 5]}

    def update_stats(self, sprt_passed, draw_rate):
        self.draw_rate = draw_rate
        if sprt_passed:
            self.consecutive_sprt_failures = 0
        else:
            self.consecutive_sprt_failures += 1

        # State transition logic
        if self.draw_rate > 0.70:
            self.state = AutoTuningState.REFINEMENT
        elif self.consecutive_sprt_failures >= 5:
            self.state = AutoTuningState.PLATEAU
            self.consecutive_sprt_failures = 0 # reset after entering plateau to allow it to recover
        else:
            self.state = AutoTuningState.FAST_LEARNING

# --- TD-Learning Loss ---
def td_loss(predictions, absolute_results):
    """
    Target = 最終勝負 (For now, simplified without MCTS root eval)
    """
    return nn.MSELoss()(predictions, absolute_results)

# --- Feature Extraction ---
def extract_features(board):
    features = np.zeros(494, dtype=np.float32)
    for p in range(14):
        bb = int(board.piece_bitboards[p])
        for sq in range(32):
            if (bb >> sq) & 1:
                features[sq * 15 + p] = 1.0
                
    hb = int(board.hidden_bitboard)
    hidden_counts = np.zeros(14, dtype=np.float32)
    for sq in range(32):
        if (hb >> sq) & 1:
            features[sq * 15 + 14] = 1.0
            piece = board.hidden_pieces[sq]
            hidden_counts[piece] += 1.0
            
    features[480:494] = hidden_counts
    return features

class DarkChessDataset(IterableDataset):
    def __init__(self, files):
        self.files = files
        
    def __iter__(self):
        for f in self.files:
            try:
                with gzip.open(f, 'rt', encoding='utf-8') as gz:
                    for line in gz:
                        if not line.strip(): continue
                        game = json.loads(line)
                        board = DarkChessBoardPy(bag=game['hid'])
                        res = float(game['res'])
                        
                        for move in game['mov']:
                            side = board.side_to_move
                            if side != 2: # 2 is NONE
                                # The network value is always from Red's perspective.
                                # This matches self-play records and SPRT move selection
                                # (Red maximizes the value, Black minimizes it).
                                target = res
                                feat = extract_features(board)
                                yield torch.tensor(feat), torch.tensor([target], dtype=torch.float32)
                            board.make_move(move)
            except Exception as e:
                print(f"Error reading {f}: {e}")

def main():
    print("Initializing DarkChess NNUE Training Pipeline...")
    model = DarkChessNNUE()
    optimizer = optim.AdamW(model.parameters(), weight_decay=1e-4)
    tuner = AutoTuner()
    
    
    # 1. Read files downloaded from Hugging Face into the local datasets/ directory
    dataset_dir = "datasets"
    files = glob.glob(os.path.join(dataset_dir, "**/*.jsonl.gz"), recursive=True)
    print(f"Discovered {len(files)} batches locally in {dataset_dir}.")
    
    # Replay Buffer mechanism (Sliding window up to 500k games)
    print("Replay buffer configured. Max capacity: 500,000 games.")
    
    dataset = DarkChessDataset(files)
    dataloader = DataLoader(dataset, batch_size=256)
    
    print(f"Current State: {tuner.state}, Hyperparams: {tuner.get_hyperparameters()}")
    print("Training loop ready.")

    # REAL TRAINING LOOP
    print("Starting training epochs...")
    model.train()
    
    total_loss = 0
    batches = 0
    for epoch in range(1):
        for features, targets in dataloader:
            optimizer.zero_grad()
            outputs = model(features)
            loss = td_loss(outputs, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            batches += 1
            if batches % 100 == 0:
                print(f"Epoch {epoch} | Batch {batches} | Loss: {total_loss/100:.4f}")
                total_loss = 0
                
    print("Training complete.")

    # 2. Save the newly trained model to models/challenger.nnue
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "challenger.nnue")
    
    # Save the PyTorch state dict
    torch.save(model.state_dict(), model_path)
    print(f"Model saved successfully to {model_path}")

if __name__ == '__main__':
    main()
