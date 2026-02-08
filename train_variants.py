"""
LSTM Forgetting Curve Variants Trainer
=======================================
Trains 5 different variations of the exponential decay forgetting curve formula
and generates comparison plots for all variants.

Variants:
1. Original: interval = -log(R) / exp(o(x))
2. Power Law: interval = -log(R) / (exp(o(x)))^α where α=1.2
3. Linear Decay: interval = -log(R) / (a*o(x) + b) where a,b are learnable
4. Sigmoid Modulated: interval = -log(R) / (exp(o(x)) * sigmoid(β*o(x)))
5. Adaptive Target: interval = -log(R(x)) / exp(o(x)) where R(x) is learnable per sample

Usage:
    python train_variants.py --dataset dsa_spaced_repetition_dataset.csv
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
from datetime import datetime
import json
from train_lstm_model import DSASequenceDataset

sns.set_style("whitegrid")


class Variant1_Original(nn.Module):
    """Original: interval = -log(R) / exp(o(x))"""
    
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Linear(32, 1)
        )
        self.log_target_recall = nn.Parameter(torch.tensor([-0.10536]))
        self.fc[-1].bias.data.fill_(-3.70)
    
    def forward(self, x, lag_times=None):
        lstm_out, _ = self.lstm(x)
        decay_param = self.fc(lstm_out[:, -1, :])
        target_recall = torch.exp(self.log_target_recall)
        intervals = -torch.log(target_recall) / torch.exp(decay_param)
        return torch.clamp(intervals, min=1.0, max=90.0)


class Variant2_PowerLaw(nn.Module):
    """Power Law: interval = -log(R) / (exp(o(x)))^α"""
    
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Linear(32, 1)
        )
        self.log_target_recall = nn.Parameter(torch.tensor([-0.10536]))
        self.alpha = nn.Parameter(torch.tensor([1.2]))  # Power law exponent
        self.fc[-1].bias.data.fill_(-3.70)
    
    def forward(self, x, lag_times=None):
        lstm_out, _ = self.lstm(x)
        decay_param = self.fc(lstm_out[:, -1, :])
        target_recall = torch.exp(self.log_target_recall)
        # Apply power law to decay rate
        intervals = -torch.log(target_recall) / (torch.exp(decay_param) ** self.alpha)
        return torch.clamp(intervals, min=1.0, max=90.0)


class Variant3_LinearDecay(nn.Module):
    """Linear Decay: interval = -log(R) / (a*o(x) + b)"""
    
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Linear(32, 1)
        )
        self.log_target_recall = nn.Parameter(torch.tensor([-0.10536]))
        # Learnable linear transformation parameters
        self.a = nn.Parameter(torch.tensor([0.1]))
        self.b = nn.Parameter(torch.tensor([0.05]))
        self.fc[-1].bias.data.fill_(-3.70)
    
    def forward(self, x, lag_times=None):
        lstm_out, _ = self.lstm(x)
        decay_param = self.fc(lstm_out[:, -1, :])
        target_recall = torch.exp(self.log_target_recall)
        # Linear transformation instead of exponential
        decay_rate = torch.abs(self.a) * decay_param + torch.abs(self.b) + 1e-6
        intervals = -torch.log(target_recall) / decay_rate
        return torch.clamp(intervals, min=1.0, max=90.0)


class Variant4_SigmoidModulated(nn.Module):
    """Sigmoid Modulated: interval = -log(R) / (exp(o(x)) * sigmoid(β*o(x)))"""
    
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Linear(32, 1)
        )
        self.log_target_recall = nn.Parameter(torch.tensor([-0.10536]))
        self.beta = nn.Parameter(torch.tensor([0.5]))  # Sigmoid modulation strength
        self.fc[-1].bias.data.fill_(-3.70)
    
    def forward(self, x, lag_times=None):
        lstm_out, _ = self.lstm(x)
        decay_param = self.fc(lstm_out[:, -1, :])
        target_recall = torch.exp(self.log_target_recall)
        # Modulate with sigmoid for smooth transitions
        modulation = torch.sigmoid(self.beta * decay_param)
        intervals = -torch.log(target_recall) / (torch.exp(decay_param) * modulation)
        return torch.clamp(intervals, min=1.0, max=90.0)


class Variant5_AdaptiveTarget(nn.Module):
    """Adaptive Target: interval = -log(R(x)) / exp(o(x)) where R(x) is predicted"""
    
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        # Decay parameter predictor
        self.fc_decay = nn.Sequential(
            nn.Linear(hidden_size, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Linear(32, 1)
        )
        # Adaptive target recall predictor
        self.fc_target = nn.Sequential(
            nn.Linear(hidden_size, 32), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Linear(32, 1), nn.Sigmoid()  # Output between 0 and 1
        )
        self.fc_decay[-1].bias.data.fill_(-3.70)
        self.fc_target[-2].bias.data.fill_(2.2)  # Bias toward 0.9
    
    def forward(self, x, lag_times=None):
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        decay_param = self.fc_decay(last_hidden)
        # Predict adaptive target recall (0.7 to 0.95 range)
        target_recall = 0.7 + 0.25 * self.fc_target(last_hidden)
        intervals = -torch.log(target_recall + 1e-6) / (torch.exp(decay_param) + 1e-6)
        return torch.clamp(intervals, min=1.0, max=90.0)


def train_variant(model, train_loader, val_loader, criterion, optimizer, 
                  num_epochs, device, variant_name, scheduler=None):
    """Train a single variant"""
    print(f"\nTraining {variant_name}...")
    model = model.to(device)
    best_val_loss = float('inf')
    best_model_state = None
    train_losses, val_losses = [], []
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for sequences, targets in train_loader:
            sequences, targets = sequences.to(device), targets.to(device)
            optimizer.zero_grad()
            predictions = model(sequences)
            loss = criterion(predictions, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for sequences, targets in val_loader:
                sequences, targets = sequences.to(device), targets.to(device)
                predictions = model(sequences)
                loss = criterion(predictions, targets)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        if scheduler:
            scheduler.step()
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch [{epoch+1:3d}/{num_epochs}] | Train: {train_loss:.4f} | Val: {val_loss:.4f}")
    
    model.load_state_dict(best_model_state)
    print(f"  Best val loss: {best_val_loss:.4f}")
    return model, train_losses, val_losses, best_val_loss


def evaluate_variant(model, test_loader, device):
    """Evaluate a variant on test set"""
    model.eval()
    all_predictions, all_targets = [], []
    
    with torch.no_grad():
        for sequences, targets in test_loader:
            sequences = sequences.to(device)
            predictions = model(sequences)
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
    
    predictions = np.concatenate(all_predictions).flatten()
    targets = np.concatenate(all_targets).flatten()
    
    mae = np.abs(predictions - targets).mean()
    rmse = np.sqrt(((predictions - targets) ** 2).mean())
    
    return mae, rmse, predictions, targets


def plot_all_variants(results, output_dir):
    """Generate comprehensive comparison plots"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Training curves comparison
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Forgetting Curve Variants - Training Comparison', fontsize=16, fontweight='bold')
    
    for idx, (name, data) in enumerate(results.items()):
        ax = axes[idx // 3, idx % 3]
        epochs = range(1, len(data['train_losses']) + 1)
        ax.plot(epochs, data['train_losses'], label='Train', linewidth=2)
        ax.plot(epochs, data['val_losses'], label='Validation', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title(f'{name}\nMAE: {data["mae"]:.3f}', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide last subplot if odd number
    if len(results) < 6:
        axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'training_curves_{timestamp}.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Saved training curves")
    plt.close()
    
    # 2. Performance metrics comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Performance Metrics Comparison', fontsize=16, fontweight='bold')
    
    names = list(results.keys())
    maes = [results[n]['mae'] for n in names]
    rmses = [results[n]['rmse'] for n in names]
    val_losses = [results[n]['best_val_loss'] for n in names]
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(names)))
    
    axes[0].bar(range(len(names)), maes, color=colors)
    axes[0].set_xticks(range(len(names)))
    axes[0].set_xticklabels(names, rotation=45, ha='right')
    axes[0].set_ylabel('MAE (days)')
    axes[0].set_title('Mean Absolute Error')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    axes[1].bar(range(len(names)), rmses, color=colors)
    axes[1].set_xticks(range(len(names)))
    axes[1].set_xticklabels(names, rotation=45, ha='right')
    axes[1].set_ylabel('RMSE (days)')
    axes[1].set_title('Root Mean Squared Error')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    axes[2].bar(range(len(names)), val_losses, color=colors)
    axes[2].set_xticks(range(len(names)))
    axes[2].set_xticklabels(names, rotation=45, ha='right')
    axes[2].set_ylabel('Loss')
    axes[2].set_title('Best Validation Loss')
    axes[2].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'metrics_comparison_{timestamp}.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Saved metrics comparison")
    plt.close()
    
    # 3. Predictions scatter plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Predictions vs Actuals - All Variants', fontsize=16, fontweight='bold')
    
    for idx, (name, data) in enumerate(results.items()):
        ax = axes[idx // 3, idx % 3]
        predictions = data['predictions']
        targets = data['targets']
        
        ax.scatter(targets, predictions, alpha=0.3, s=10)
        max_val = max(targets.max(), predictions.max())
        ax.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Perfect')
        ax.set_xlabel('Actual Interval (days)')
        ax.set_ylabel('Predicted Interval (days)')
        ax.set_title(f'{name}\nMAE: {data["mae"]:.3f}', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    if len(results) < 6:
        axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'predictions_scatter_{timestamp}.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Saved predictions scatter")
    plt.close()
    
    # 4. Error distributions
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Prediction Error Distributions', fontsize=16, fontweight='bold')
    
    for idx, (name, data) in enumerate(results.items()):
        ax = axes[idx // 3, idx % 3]
        errors = data['predictions'] - data['targets']
        
        ax.hist(errors, bins=50, edgecolor='black', alpha=0.7, color=colors[idx])
        ax.axvline(0, color='r', linestyle='--', linewidth=2, label='Zero Error')
        ax.axvline(errors.mean(), color='g', linestyle='--', linewidth=2, 
                  label=f'Mean: {errors.mean():.2f}')
        ax.set_xlabel('Error (days)')
        ax.set_ylabel('Frequency')
        ax.set_title(f'{name}', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    if len(results) < 6:
        axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'error_distributions_{timestamp}.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Saved error distributions")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Train 5 forgetting curve variants')
    parser.add_argument('--dataset', type=str, required=True, help='Path to CSV dataset')
    parser.add_argument('--epochs', type=int, default=50, help='Epochs per variant (default: 50)')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size (default: 256)')
    parser.add_argument('--output-dir', type=str, default='variant_results', 
                       help='Output directory (default: variant_results)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 
                         'cuda' if torch.cuda.is_available() else 'cpu')
    
    print("="*70)
    print("FORGETTING CURVE VARIANTS TRAINER")
    print("="*70)
    print(f"Dataset: {args.dataset}")
    print(f"Device: {device}")
    print(f"Epochs per variant: {args.epochs}")
    
    # Load and prepare data
    df = pd.read_csv(args.dataset)
    train_df, test_df = train_test_split(df, test_size=0.15, random_state=args.seed)
    train_df, val_df = train_test_split(train_df, test_size=0.15, random_state=args.seed)
    
    train_dataset = DSASequenceDataset(train_df, max_seq_length=40, fit_scaler=True)
    val_dataset = DSASequenceDataset(val_df, max_seq_length=40, 
                                     scaler=train_dataset.scaler, fit_scaler=False)
    test_dataset = DSASequenceDataset(test_df, max_seq_length=40,
                                      scaler=train_dataset.scaler, fit_scaler=False)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    input_size = train_dataset.sequences.shape[2]
    
    # Define variants
    variants = {
        'V1: Original': Variant1_Original,
        'V2: Power Law': Variant2_PowerLaw,
        'V3: Linear Decay': Variant3_LinearDecay,
        'V4: Sigmoid Mod': Variant4_SigmoidModulated,
        'V5: Adaptive Target': Variant5_AdaptiveTarget
    }
    
    results = {}
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Train each variant
    for name, ModelClass in variants.items():
        print(f"\n{'='*70}")
        print(f"Training: {name}")
        print(f"{'='*70}")
        
        model = ModelClass(input_size)
        criterion = nn.HuberLoss(delta=1.0)
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )
        
        model, train_losses, val_losses, best_val_loss = train_variant(
            model, train_loader, val_loader, criterion, optimizer,
            args.epochs, device, name, scheduler
        )
        
        mae, rmse, predictions, targets = evaluate_variant(model, test_loader, device)
        
        print(f"\n  Test Results:")
        print(f"    MAE:  {mae:.4f} days")
        print(f"    RMSE: {rmse:.4f} days")
        
        results[name] = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_val_loss': best_val_loss,
            'mae': mae,
            'rmse': rmse,
            'predictions': predictions,
            'targets': targets
        }
    
    # Generate comparison plots
    print(f"\n{'='*70}")
    print("Generating Comparison Plots")
    print(f"{'='*70}")
    plot_all_variants(results, args.output_dir)
    
    # Save summary
    summary = {name: {'mae': float(data['mae']), 'rmse': float(data['rmse']),
                      'best_val_loss': float(data['best_val_loss'])}
               for name, data in results.items()}
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(os.path.join(args.output_dir, f'summary_{timestamp}.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print final summary
    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print(f"{'='*70}")
    print(f"{'Variant':<25} {'MAE':<12} {'RMSE':<12} {'Val Loss':<12}")
    print("-"*70)
    for name, data in results.items():
        print(f"{name:<25} {data['mae']:<12.4f} {data['rmse']:<12.4f} {data['best_val_loss']:<12.4f}")
    
    best_variant = min(results.items(), key=lambda x: x[1]['mae'])
    print(f"\n🏆 Best Variant: {best_variant[0]} (MAE: {best_variant[1]['mae']:.4f})")
    print(f"\n✓ All results saved to: {args.output_dir}/")
    print("="*70)


if __name__ == '__main__':
    main()
