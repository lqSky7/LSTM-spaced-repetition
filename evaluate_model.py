"""
Model Evaluation Script for LSTM Spaced Repetition
===================================================
Evaluates trained LSTM models using k-means clustering analysis and 
comprehensive visualization across different datasets.

Features:
- K-means clustering to identify student/problem patterns
- Performance metrics per cluster
- Comparative analysis between datasets
- Rich visualization suite

Usage:
    python evaluate_model.py --model models/exp-decay_lstm_20251226_122714.pt \
                             --datasets dsa_spaced_repetition_dataset.csv dsa_synthetic_dataset.csv
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
from datetime import datetime
import json
from train_lstm_model import (
    StandardLSTM, 
    LSTMWithExponentialDecay, 
    DSASequenceDataset
)

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


class ModelEvaluator:
    """Comprehensive model evaluation with k-means clustering analysis"""
    
    def __init__(self, model_path, output_dir='results'):
        """
        Args:
            model_path: Path to saved model checkpoint
            output_dir: Directory to save results and plots
        """
        self.model_path = model_path
        self.output_dir = output_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create output directories
        self.results_dir = os.path.join(output_dir, 'metrics')
        self.plots_dir = os.path.join(output_dir, 'plots')
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)
        
        # Load model
        self._load_model()
        
    def _load_model(self):
        """Load trained model from checkpoint"""
        print(f"Loading model from {self.model_path}...")
        
        checkpoint = torch.load(self.model_path, map_location='cpu', weights_only=False)
        
        self.model_type = checkpoint['model_type']
        input_size = checkpoint['input_size']
        hidden_size = checkpoint['hidden_size']
        num_layers = checkpoint['num_layers']
        self.scaler = checkpoint['scaler']
        self.feature_cols = checkpoint['feature_cols']
        
        # Create model
        if self.model_type == 'standard':
            self.model = StandardLSTM(input_size, hidden_size, num_layers)
        else:
            self.model = LSTMWithExponentialDecay(input_size, hidden_size, num_layers)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Device selection
        if torch.backends.mps.is_available():
            self.device = torch.device('mps')
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        
        self.model = self.model.to(self.device)
        print(f"✓ Model loaded successfully (Type: {self.model_type}, Device: {self.device})")
        
    def evaluate_dataset(self, dataset_path, dataset_name, n_clusters=5):
        """
        Evaluate model on a dataset with k-means clustering analysis
        
        Args:
            dataset_path: Path to CSV dataset
            dataset_name: Name for labeling (e.g., 'real', 'synthetic')
            n_clusters: Number of clusters for k-means
            
        Returns:
            Dictionary with evaluation results
        """
        print(f"\n{'='*70}")
        print(f"Evaluating: {dataset_name}")
        print(f"{'='*70}")
        
        # Load dataset
        df = pd.read_csv(dataset_path)
        print(f"Loaded {len(df)} records")
        
        # Handle mistake tags
        if 'mistake_tags' in df.columns and df['mistake_tags'].dtype == 'object':
            df['num_mistakes'] = df['mistake_tags'].apply(
                lambda x: len(x.split(',')) if isinstance(x, str) and x else 0
            )
        
        # Ensure dataset only uses the features the model was trained on
        # Add any missing features with default values
        for col in self.feature_cols:
            if col not in df.columns:
                df[col] = 0  # Default value for missing features
        
        # Create dataset with the same features the model was trained on
        dataset = DSASequenceDataset(
            df, max_seq_length=40, scaler=self.scaler, fit_scaler=False,
            feature_cols=self.feature_cols
        )
        
        # Get predictions
        predictions, actuals = self._get_predictions(dataset)
        
        # Calculate base metrics
        mae = np.abs(predictions - actuals).mean()
        rmse = np.sqrt(((predictions - actuals) ** 2).mean())
        mape = (np.abs((actuals - predictions) / (actuals + 1e-6)).mean() * 100)
        
        print(f"\n📊 Overall Metrics:")
        print(f"   MAE:  {mae:.4f} days")
        print(f"   RMSE: {rmse:.4f} days")
        print(f"   MAPE: {mape:.2f}%")
        
        # Prepare features for clustering
        # Extract last observation for each sequence
        features_for_clustering = []
        for i in range(len(df)):
            if i < len(dataset.sequences):
                # Get last non-zero features from sequence
                seq = dataset.sequences[i]
                non_zero_rows = np.any(seq != 0, axis=1)
                if non_zero_rows.any():
                    last_features = seq[non_zero_rows][-1]
                    features_for_clustering.append(last_features)
        
        features_for_clustering = np.array(features_for_clustering[:len(predictions)])
        
        # K-means clustering
        print(f"\n🔍 K-Means Clustering (k={n_clusters})...")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(features_for_clustering)
        
        # Clustering quality metrics
        silhouette = silhouette_score(features_for_clustering, cluster_labels)
        davies_bouldin = davies_bouldin_score(features_for_clustering, cluster_labels)
        
        print(f"   Silhouette Score: {silhouette:.4f} (higher is better, range: -1 to 1)")
        print(f"   Davies-Bouldin Index: {davies_bouldin:.4f} (lower is better)")
        
        # Analyze performance per cluster
        cluster_metrics = self._analyze_clusters(
            predictions, actuals, cluster_labels, n_clusters
        )
        
        # Store results
        results = {
            'dataset_name': dataset_name,
            'dataset_path': dataset_path,
            'n_records': len(df),
            'n_predictions': len(predictions),
            'overall_metrics': {
                'mae': float(mae),
                'rmse': float(rmse),
                'mape': float(mape),
                'mean_predicted': float(predictions.mean()),
                'mean_actual': float(actuals.mean())
            },
            'clustering': {
                'n_clusters': n_clusters,
                'silhouette_score': float(silhouette),
                'davies_bouldin_index': float(davies_bouldin),
                'cluster_metrics': cluster_metrics
            },
            'predictions': predictions.tolist(),
            'actuals': actuals.tolist(),
            'cluster_labels': cluster_labels.tolist(),
            'cluster_centers': kmeans.cluster_centers_.tolist()
        }
        
        # Generate plots
        self._generate_plots(results, dataset_name)
        
        return results
    
    def _get_predictions(self, dataset):
        """Get model predictions for dataset"""
        predictions = []
        actuals = []
        
        with torch.no_grad():
            for i in range(len(dataset)):
                sequence, target = dataset[i]
                sequence = sequence.unsqueeze(0).to(self.device)
                
                pred = self.model(sequence)
                predictions.append(pred.cpu().item())
                actuals.append(target.item())
        
        return np.array(predictions), np.array(actuals)
    
    def _analyze_clusters(self, predictions, actuals, cluster_labels, n_clusters):
        """Analyze model performance per cluster"""
        print(f"\n📈 Per-Cluster Performance:")
        print(f"   {'Cluster':<10} {'Size':<8} {'MAE':<10} {'RMSE':<10} {'Mean Pred':<12} {'Mean Actual':<12}")
        print(f"   {'-'*72}")
        
        cluster_metrics = {}
        
        for cluster_id in range(n_clusters):
            mask = cluster_labels == cluster_id
            cluster_preds = predictions[mask]
            cluster_actuals = actuals[mask]
            
            if len(cluster_preds) == 0:
                continue
            
            mae = np.abs(cluster_preds - cluster_actuals).mean()
            rmse = np.sqrt(((cluster_preds - cluster_actuals) ** 2).mean())
            
            cluster_metrics[int(cluster_id)] = {
                'size': int(mask.sum()),
                'mae': float(mae),
                'rmse': float(rmse),
                'mean_predicted': float(cluster_preds.mean()),
                'mean_actual': float(cluster_actuals.mean())
            }
            
            print(f"   {cluster_id:<10} {mask.sum():<8} {mae:<10.4f} {rmse:<10.4f} "
                  f"{cluster_preds.mean():<12.2f} {cluster_actuals.mean():<12.2f}")
        
        return cluster_metrics
    
    def _generate_plots(self, results, dataset_name):
        """Generate comprehensive visualization plots"""
        print(f"\n📊 Generating plots for {dataset_name}...")
        
        predictions = np.array(results['predictions'])
        actuals = np.array(results['actuals'])
        cluster_labels = np.array(results['cluster_labels'])
        n_clusters = results['clustering']['n_clusters']
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 12))
        
        # 1. Predictions vs Actuals Scatter
        ax1 = plt.subplot(2, 3, 1)
        ax1.scatter(actuals, predictions, alpha=0.5, s=10)
        max_val = max(actuals.max(), predictions.max())
        ax1.plot([0, max_val], [0, max_val], 'r--', label='Perfect Prediction')
        ax1.set_xlabel('Actual Interval (days)')
        ax1.set_ylabel('Predicted Interval (days)')
        ax1.set_title(f'Predictions vs Actuals - {dataset_name}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Error Distribution
        ax2 = plt.subplot(2, 3, 2)
        errors = predictions - actuals
        ax2.hist(errors, bins=50, edgecolor='black', alpha=0.7)
        ax2.axvline(0, color='r', linestyle='--', label='Zero Error')
        ax2.axvline(errors.mean(), color='g', linestyle='--', 
                    label=f'Mean Error: {errors.mean():.2f}')
        ax2.set_xlabel('Prediction Error (days)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Error Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Cluster-based Predictions vs Actuals
        ax3 = plt.subplot(2, 3, 3)
        colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))
        for cluster_id in range(n_clusters):
            mask = cluster_labels == cluster_id
            if mask.sum() > 0:
                ax3.scatter(actuals[mask], predictions[mask], 
                           alpha=0.6, s=20, c=[colors[cluster_id]], 
                           label=f'Cluster {cluster_id}')
        max_val = max(actuals.max(), predictions.max())
        ax3.plot([0, max_val], [0, max_val], 'r--', alpha=0.5)
        ax3.set_xlabel('Actual Interval (days)')
        ax3.set_ylabel('Predicted Interval (days)')
        ax3.set_title('Predictions by Cluster')
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3)
        
        # 4. MAE per Cluster
        ax4 = plt.subplot(2, 3, 4)
        cluster_ids = list(results['clustering']['cluster_metrics'].keys())
        maes = [results['clustering']['cluster_metrics'][cid]['mae'] 
                for cid in cluster_ids]
        ax4.bar(cluster_ids, maes, color=colors[:len(cluster_ids)])
        ax4.set_xlabel('Cluster ID')
        ax4.set_ylabel('MAE (days)')
        ax4.set_title('MAE per Cluster')
        ax4.grid(True, alpha=0.3, axis='y')
        
        # 5. Cluster Sizes
        ax5 = plt.subplot(2, 3, 5)
        sizes = [results['clustering']['cluster_metrics'][cid]['size'] 
                 for cid in cluster_ids]
        ax5.bar(cluster_ids, sizes, color=colors[:len(cluster_ids)])
        ax5.set_xlabel('Cluster ID')
        ax5.set_ylabel('Number of Samples')
        ax5.set_title('Cluster Sizes')
        ax5.grid(True, alpha=0.3, axis='y')
        
        # 6. Predicted vs Actual Means per Cluster
        ax6 = plt.subplot(2, 3, 6)
        x = np.arange(len(cluster_ids))
        width = 0.35
        pred_means = [results['clustering']['cluster_metrics'][cid]['mean_predicted'] 
                      for cid in cluster_ids]
        actual_means = [results['clustering']['cluster_metrics'][cid]['mean_actual'] 
                        for cid in cluster_ids]
        ax6.bar(x - width/2, pred_means, width, label='Predicted', alpha=0.8)
        ax6.bar(x + width/2, actual_means, width, label='Actual', alpha=0.8)
        ax6.set_xlabel('Cluster ID')
        ax6.set_ylabel('Mean Interval (days)')
        ax6.set_title('Mean Intervals per Cluster')
        ax6.set_xticks(x)
        ax6.set_xticklabels(cluster_ids)
        ax6.legend()
        ax6.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        # Save plot
        plot_filename = f"{dataset_name}_evaluation_{self.timestamp}.png"
        plot_path = os.path.join(self.plots_dir, plot_filename)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"   ✓ Saved plot: {plot_path}")
        plt.close()
    
    def compare_datasets(self, results_list):
        """Generate comparison plots between datasets"""
        print(f"\n{'='*70}")
        print("Generating Dataset Comparison")
        print(f"{'='*70}")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. MAE Comparison
        ax = axes[0, 0]
        dataset_names = [r['dataset_name'] for r in results_list]
        maes = [r['overall_metrics']['mae'] for r in results_list]
        ax.bar(dataset_names, maes, color=['#3498db', '#e74c3c'])
        ax.set_ylabel('MAE (days)')
        ax.set_title('Mean Absolute Error Comparison')
        ax.grid(True, alpha=0.3, axis='y')
        
        # 2. RMSE Comparison
        ax = axes[0, 1]
        rmses = [r['overall_metrics']['rmse'] for r in results_list]
        ax.bar(dataset_names, rmses, color=['#3498db', '#e74c3c'])
        ax.set_ylabel('RMSE (days)')
        ax.set_title('Root Mean Squared Error Comparison')
        ax.grid(True, alpha=0.3, axis='y')
        
        # 3. Prediction Distribution
        ax = axes[1, 0]
        for i, result in enumerate(results_list):
            predictions = np.array(result['predictions'])
            ax.hist(predictions, bins=30, alpha=0.6, 
                   label=result['dataset_name'], edgecolor='black')
        ax.set_xlabel('Predicted Interval (days)')
        ax.set_ylabel('Frequency')
        ax.set_title('Prediction Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Clustering Quality
        ax = axes[1, 1]
        x = np.arange(len(dataset_names))
        width = 0.35
        silhouettes = [r['clustering']['silhouette_score'] for r in results_list]
        ax.bar(x, silhouettes, width, label='Silhouette Score', alpha=0.8)
        ax.set_ylabel('Score')
        ax.set_title('Clustering Quality (Silhouette Score)')
        ax.set_xticks(x)
        ax.set_xticklabels(dataset_names)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        # Save comparison plot
        plot_filename = f"dataset_comparison_{self.timestamp}.png"
        plot_path = os.path.join(self.plots_dir, plot_filename)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved comparison plot: {plot_path}")
        plt.close()
        
    def save_results(self, results_list):
        """Save evaluation results to JSON"""
        output = {
            'model_path': self.model_path,
            'model_type': self.model_type,
            'timestamp': self.timestamp,
            'evaluations': results_list
        }
        
        results_filename = f"evaluation_results_{self.timestamp}.json"
        results_path = os.path.join(self.results_dir, results_filename)
        
        with open(results_path, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"\n✓ Results saved to: {results_path}")
        
        # Also save a summary CSV
        summary_data = []
        for result in results_list:
            summary_data.append({
                'Dataset': result['dataset_name'],
                'N_Records': result['n_records'],
                'MAE': result['overall_metrics']['mae'],
                'RMSE': result['overall_metrics']['rmse'],
                'MAPE': result['overall_metrics']['mape'],
                'Mean_Predicted': result['overall_metrics']['mean_predicted'],
                'Mean_Actual': result['overall_metrics']['mean_actual'],
                'Silhouette_Score': result['clustering']['silhouette_score'],
                'Davies_Bouldin': result['clustering']['davies_bouldin_index']
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_csv = os.path.join(self.results_dir, f"summary_{self.timestamp}.csv")
        summary_df.to_csv(summary_csv, index=False)
        print(f"✓ Summary saved to: {summary_csv}")


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate LSTM model with k-means clustering analysis'
    )
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--datasets', type=str, nargs='+', required=True,
                       help='Paths to datasets to evaluate')
    parser.add_argument('--n-clusters', type=int, default=5,
                       help='Number of clusters for k-means (default: 5)')
    parser.add_argument('--output-dir', type=str, default='results',
                       help='Output directory for results (default: results)')
    
    args = parser.parse_args()
    
    print("="*70)
    print("LSTM SPACED REPETITION MODEL EVALUATION")
    print("="*70)
    print(f"Model: {args.model}")
    print(f"Datasets: {', '.join(args.datasets)}")
    print(f"K-means clusters: {args.n_clusters}")
    print(f"Output directory: {args.output_dir}")
    
    # Create evaluator
    evaluator = ModelEvaluator(args.model, args.output_dir)
    
    # Evaluate each dataset
    all_results = []
    for dataset_path in args.datasets:
        # Extract dataset name from filename
        dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
        dataset_name = dataset_name.replace('_', ' ').title()
        
        results = evaluator.evaluate_dataset(
            dataset_path, 
            dataset_name, 
            n_clusters=args.n_clusters
        )
        all_results.append(results)
    
    # Compare datasets if multiple
    if len(all_results) > 1:
        evaluator.compare_datasets(all_results)
    
    # Save all results
    evaluator.save_results(all_results)
    
    print("\n" + "="*70)
    print("✓ EVALUATION COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(f"\nResults saved in: {args.output_dir}/")
    print(f"  - Metrics: {os.path.join(args.output_dir, 'metrics')}/")
    print(f"  - Plots:   {os.path.join(args.output_dir, 'plots')}/")


if __name__ == '__main__':
    main()
