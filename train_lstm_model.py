"""
LSTM Training Script for DSA Spaced Repetition
================================================
Implements the LSTM architecture from "Modeling Spaced Repetition with LSTMs" 
(Pokrywka et al., 2023) with exponential decay for predicting review intervals.

Supports: MPS (Apple Silicon), and CPU

Two models:
1. Standard LSTM
2. LSTM with Exponential Decay (better for spaced repetition)

Usage:
    python train_lstm_model.py --dataset dsa_spaced_repetition_dataset.csv
    python train_lstm_model.py --dataset dsa_synthetic_dataset.csv --model-type exp-decay
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import argparse
import os
from datetime import datetime
import json


class DSASequenceDataset(Dataset):
    """
    Dataset for LSTM-based spaced repetition prediction.
    Maintains sequence history for each user-problem pair.
    """

    def __init__(self, df, max_seq_length=40, scaler=None, fit_scaler=False, feature_cols=None):
        """
        Args:
            df: DataFrame with learning records
            max_seq_length: Maximum sequence length (pad/truncate)
            scaler: Optional StandardScaler for feature normalization
            fit_scaler: Whether to fit the scaler on this data
            feature_cols: Optional list of feature columns to use (if None, auto-detect)
        """
        self.max_seq_length = max_seq_length
        self.sequences = []
        self.targets = []

        # Feature columns (excluding target and identifiers)
        if feature_cols is not None:
            # Use provided feature columns
            self.feature_cols = feature_cols
        else:
            # Auto-detect feature columns
            self.feature_cols = [
                'difficulty', 'category', 
                'attempt_number', 'days_since_last_attempt',
                'outcome',  # SUCCESS/FAILURE - critical for interval prediction!
                'num_tries', 'time_spent_minutes'
            ]

            # Add optional features if they exist
            optional_features = ['time_complexity_class', 'code_lines', 'success_streak']
            for feat in optional_features:
                if feat in df.columns:
                    self.feature_cols.append(feat)

        # Initialize or use provided scaler
        if fit_scaler:
            self.scaler = StandardScaler()
        else:
            self.scaler = scaler

        # Process data into sequences
        self._build_sequences(df, fit_scaler)

    def _build_sequences(self, df, fit_scaler):
        """Build sequences grouped by user and problem"""

        # Sort by user, problem, and timestamp
        df = df.sort_values(['user_id', 'problem_id', 'timestamp'])

        # Group by user and problem
        grouped = df.groupby(['user_id', 'problem_id'])

        all_features = []
        all_targets = []
        all_lags = []

        for (user_id, problem_id), group in grouped:
            group = group.sort_values('timestamp')

            # Extract features and target
            features = group[self.feature_cols].values
            targets = group['review_interval'].values

            # Days since last attempt (lag time for exponential decay)
            lags = group['days_since_last_attempt'].values

            # Create sequences of increasing length
            for i in range(1, len(features) + 1):
                seq_features = features[:i]
                seq_target = targets[i-1]  # Predict interval for last attempt
                seq_lag = lags[i-1]

                all_features.append(seq_features)
                all_targets.append(seq_target)
                all_lags.append(seq_lag)

        # Fit scaler on all features if needed
        if fit_scaler:
            all_features_flat = np.vstack(all_features)
            self.scaler.fit(all_features_flat)

        # Normalize and pad sequences
        for features, target, lag in zip(all_features, all_targets, all_lags):
            # Normalize features
            if self.scaler is not None:
                features = self.scaler.transform(features)

            # Pad or truncate sequence
            seq_len = len(features)
            if seq_len < self.max_seq_length:
                # Pad with zeros
                padding = np.zeros((self.max_seq_length - seq_len, features.shape[1]))
                features = np.vstack([padding, features])
            else:
                # Take last max_seq_length items
                features = features[-self.max_seq_length:]

            self.sequences.append(features.astype(np.float32))
            self.targets.append(target)

        self.sequences = np.array(self.sequences)
        self.targets = np.array(self.targets, dtype=np.float32)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.sequences[idx]),
            torch.FloatTensor([self.targets[idx]])
        )


class Attention(nn.Module):
    """Attention mechanism to focus on important parts of sequence"""
    
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attention = nn.Linear(hidden_size, 1)
    
    def forward(self, lstm_output):
        # lstm_output shape: (batch, seq_len, hidden_size)
        # Compute attention scores
        scores = self.attention(lstm_output)  # (batch, seq_len, 1)
        weights = torch.softmax(scores, dim=1)  # (batch, seq_len, 1)
        
        # Apply attention weights
        context = torch.sum(weights * lstm_output, dim=1)  # (batch, hidden_size)
        return context, weights


class StandardLSTM(nn.Module):
    """Simplified LSTM model for sequence prediction"""

    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.2):
        super(StandardLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Simpler FC network with batch norm
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        # Initialize to predict around mean interval (4.7 days)
        self.fc[-1].bias.data.fill_(4.7)

    def forward(self, x):
        # x shape: (batch, seq_len, features)
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use last hidden state
        last_hidden = lstm_out[:, -1, :]

        # Use last hidden state
        last_hidden = lstm_out[:, -1, :]

        # Predict interval directly
        output = self.fc(last_hidden)
        
        # Clamp to reasonable range (1-90 days)
        output = torch.clamp(output, min=1.0, max=90.0)

        return output


class LSTMWithExponentialDecay(nn.Module):
    """Simplified LSTM with exponential decay"""

    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.2):
        super(LSTMWithExponentialDecay, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Output layer predicts decay rate parameter
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        # Learnable target recall probability (starts at 0.9)
        self.log_target_recall = nn.Parameter(torch.tensor([-0.10536]))  # log(0.9)

        # Initialize bias
        self.fc[-1].bias.data.fill_(-3.70)

    def forward(self, x, lag_times=None):
        """
        Args:
            x: Input sequences (batch, seq_len, features)
            lag_times: Days since last attempt (batch,) - used during inference

        Returns:
            Predicted review intervals
        """
        lstm_out, (h_n, c_n) = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]

        # LSTM outputs decay rate parameter o(x)
        decay_param = self.fc(last_hidden)

        # Convert to review interval using exponential decay inverse
        # For training: predict interval directly from decay parameter
        # interval = -log(target_recall) / exp(o(x))
        target_recall = torch.exp(self.log_target_recall)
        intervals = -torch.log(target_recall) / torch.exp(decay_param)

        # Clamp to reasonable range (1-90 days)
        intervals = torch.clamp(intervals, min=1.0, max=90.0)

        return intervals


def macro_avg_mae(predictions, targets, outcomes):
    """
    Compute Macro-Average MAE (primary metric from paper).
    Separately calculates MAE for successful and failed attempts.
    """
    successful_mask = outcomes == 1
    failed_mask = outcomes == 0

    if successful_mask.sum() > 0:
        mae_success = torch.abs(predictions[successful_mask] - targets[successful_mask]).mean()
    else:
        mae_success = torch.tensor(0.0)

    if failed_mask.sum() > 0:
        mae_failed = torch.abs(predictions[failed_mask] - targets[failed_mask]).mean()
    else:
        mae_failed = torch.tensor(0.0)

    return (mae_success + mae_failed) / 2


def train_model(model, train_loader, val_loader, criterion, optimizer, 
                num_epochs=50, device='cpu', model_name='model', scheduler=None):
    """Train the LSTM model"""

    print(f"\nTraining {model_name}...")
    print(f"Device: {device}")
    print(f"Epochs: {num_epochs}")
    print("-" * 60)

    model = model.to(device)
    best_val_loss = float('inf')
    best_model_state = None
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0

        for batch_idx, (sequences, targets) in enumerate(train_loader):
            sequences = sequences.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            # Forward pass
            predictions = model(sequences)
            loss = criterion(predictions, targets)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Validation phase
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for sequences, targets in val_loader:
                sequences = sequences.to(device)
                targets = targets.to(device)

                predictions = model(sequences)
                loss = criterion(predictions, targets)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # Update learning rate (CosineAnnealing updates every epoch)
        if scheduler is not None:
            scheduler.step()

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()

        # Print progress
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1:3d}/{num_epochs}] | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f}")

    # Load best model
    model.load_state_dict(best_model_state)

    print(f"\nTraining completed! Best validation loss: {best_val_loss:.4f}")

    return model, train_losses, val_losses


def evaluate_model(model, test_loader, device='cpu'):
    """Evaluate model on test set"""

    model.eval()
    model = model.to(device)

    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for sequences, targets in test_loader:
            sequences = sequences.to(device)
            predictions = model(sequences)

            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    predictions = np.concatenate(all_predictions).flatten()
    targets = np.concatenate(all_targets).flatten()

    # Calculate metrics
    mae = np.abs(predictions - targets).mean()
    rmse = np.sqrt(((predictions - targets) ** 2).mean())

    print("\n" + "=" * 60)
    print("TEST SET EVALUATION")
    print("=" * 60)
    print(f"MAE:  {mae:.4f} days")
    print(f"RMSE: {rmse:.4f} days")
    print(f"Mean Predicted Interval: {predictions.mean():.2f} days")
    print(f"Mean Actual Interval: {targets.mean():.2f} days")

    return mae, rmse, predictions, targets


def main():
    parser = argparse.ArgumentParser(description='Train LSTM for DSA spaced repetition')
    parser.add_argument('--dataset', type=str, required=True,
                       help='Path to CSV dataset')
    parser.add_argument('--model-type', type=str, default='standard',
                       choices=['standard', 'exp-decay'],
                       help='Model type: standard or exp-decay (default: standard)')
    parser.add_argument('--hidden-size', type=int, default=128,
                       help='LSTM hidden size (default: 128)')
    parser.add_argument('--num-layers', type=int, default=2,
                       help='Number of LSTM layers (default: 2)')
    parser.add_argument('--batch-size', type=int, default=256,
                       help='Batch size (default: 256)')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs (default: 50)')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate (default: 0.001)')
    parser.add_argument('--max-seq-length', type=int, default=40,
                       help='Maximum sequence length (default: 40)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--output-dir', type=str, default='models',
                       help='Output directory for saved models (default: models)')

    args = parser.parse_args()

    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Device configuration - supports CUDA, Apple Silicon MPS, and CPU
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("Using NVIDIA GPU (CUDA)")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using Apple Silicon GPU (Metal)")
    else:
        device = torch.device('cpu')
        print("Using CPU")

    print("=" * 60)
    print("DSA SPACED REPETITION LSTM TRAINER")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Model Type: {args.model_type}")
    print(f"Device: {device}")

    # Load data
    print(f"\nLoading dataset...")
    df = pd.read_csv(args.dataset)
    print(f"Loaded {len(df)} records")

    # Split data
    train_df, test_df = train_test_split(df, test_size=0.15, random_state=args.seed)
    train_df, val_df = train_test_split(train_df, test_size=0.15, random_state=args.seed)

    print(f"Train set: {len(train_df)} records")
    print(f"Val set: {len(val_df)} records")
    print(f"Test set: {len(test_df)} records")

    # Create datasets
    print("\nBuilding sequences...")
    train_dataset = DSASequenceDataset(
        train_df, max_seq_length=args.max_seq_length, fit_scaler=True
    )
    val_dataset = DSASequenceDataset(
        val_df, max_seq_length=args.max_seq_length, 
        scaler=train_dataset.scaler, fit_scaler=False
    )
    test_dataset = DSASequenceDataset(
        test_df, max_seq_length=args.max_seq_length,
        scaler=train_dataset.scaler, fit_scaler=False
    )

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, 
                           shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=0)

    # Get input size
    input_size = train_dataset.sequences.shape[2]
    print(f"Input feature size: {input_size}")

    # Create model
    if args.model_type == 'standard':
        model = StandardLSTM(
            input_size=input_size,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers
        )
        model_name = "Standard LSTM"
    else:
        model = LSTMWithExponentialDecay(
            input_size=input_size,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers
        )
        model_name = "LSTM with Exponential Decay"

    print(f"\nModel: {model_name}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Define loss and optimizer (Huber loss is robust to outliers)
    criterion = nn.HuberLoss(delta=1.0)  # Smooth L1 loss
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # Cosine annealing with warm restarts - better for escaping plateaus
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )

    # Train model
    model, train_losses, val_losses = train_model(
        model, train_loader, val_loader, criterion, optimizer,
        num_epochs=args.epochs, device=device, model_name=model_name,
        scheduler=scheduler
    )

    # Evaluate on test set
    mae, rmse, predictions, targets = evaluate_model(model, test_loader, device)

    # Save model
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"{args.model_type}_lstm_{timestamp}.pt"
    model_path = os.path.join(args.output_dir, model_filename)

    torch.save({
        'model_state_dict': model.state_dict(),
        'model_type': args.model_type,
        'input_size': input_size,
        'hidden_size': args.hidden_size,
        'num_layers': args.num_layers,
        'scaler': train_dataset.scaler,
        'feature_cols': train_dataset.feature_cols,
        'test_mae': mae,
        'test_rmse': rmse,
        'args': vars(args)
    }, model_path)

    print(f"\n✓ Model saved to: {model_path}")

    # Save training history
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'test_mae': float(mae),
        'test_rmse': float(rmse)
    }

    history_path = os.path.join(args.output_dir, f"{args.model_type}_history_{timestamp}.json")
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    print(f"✓ Training history saved to: {history_path}")
    print("\n" + "=" * 60)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == '__main__':
    main()