"""
Main Training Script
Orchestrates the complete training pipeline for bucket-based prediction engine
"""

import argparse
import sys
import os
from sklearn.model_selection import train_test_split
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader import EventLogProcessor, PrefixBucketer, ActivityEncoder
from outcome_models import OutcomePredictor
from bucket_models import BucketEnsemble


def train_phase1_outcome(processor, activity_encoder, output_dir, max_prefix_length=5):
    """
    Phase 1: Train outcome prediction models
    
    Args:
        processor: EventLogProcessor instance
        activity_encoder: ActivityEncoder instance
        output_dir: Directory to save models
        max_prefix_length: Maximum prefix length for outcome prediction
    """
    print("\n" + "="*70)
    print("PHASE 1: OUTCOME PREDICTION")
    print("="*70)
    
    # Generate prefixes
    prefix_samples = processor.generate_prefixes()
    
    # Train outcome predictor
    outcome_predictor = OutcomePredictor(activity_encoder)
    
    # Prepare data
    X, y = outcome_predictor.prepare_data(prefix_samples, max_prefix_length=max_prefix_length)
    
    # Split data (stratified by case to avoid leakage)
    # Important: Split by case_id, not by individual prefixes!
    case_ids = np.array([s['case_id'] for s in prefix_samples if s['prefix_length'] <= max_prefix_length])
    unique_cases = np.unique(case_ids)
    
    train_cases, test_cases = train_test_split(unique_cases, test_size=0.2, random_state=42)
    
    # Get indices for train/test
    train_mask = np.isin(case_ids, train_cases)
    test_mask = np.isin(case_ids, test_cases)
    
    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]
    
    # Further split train into train/val
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42, stratify=y_train
    )
    
    # Train
    outcome_predictor.train(X_train, y_train, X_val, y_val)
    
    # Evaluate
    outcome_predictor.evaluate(X_test, y_test)
    
    # Save
    outcome_predictor.save(os.path.join(output_dir, 'outcome'))
    
    return outcome_predictor


def train_phase2_buckets(processor, activity_encoder, output_dir, max_bucket=10, epochs=10):
    """
    Phase 2: Train bucket-based sequence models
    
    Args:
        processor: EventLogProcessor instance
        activity_encoder: ActivityEncoder instance
        output_dir: Directory to save models
        max_bucket: Maximum bucket size
        epochs: Training epochs
    """
    print("\n" + "="*70)
    print("PHASE 2: BUCKET-BASED SEQUENCE MODELS")
    print("="*70)
    
    # Generate prefixes
    prefix_samples = processor.generate_prefixes()
    
    # Bucket prefixes
    bucketer = PrefixBucketer(max_bucket=max_bucket)
    buckets = bucketer.bucket_prefixes(prefix_samples)
    
    # Initialize bucket ensemble
    ensemble = BucketEnsemble(vocab_size=activity_encoder.vocab_size, max_bucket=max_bucket)
    
    # Prepare bucket data
    bucket_data = ensemble.prepare_bucket_data(buckets, activity_encoder)
    
    # Train all buckets
    ensemble.train_all_buckets(bucket_data, epochs=epochs, batch_size=64)
    
    # Evaluate all buckets
    results = ensemble.evaluate_all_buckets(bucket_data)
    
    # Save
    ensemble.save(os.path.join(output_dir, 'buckets'))
    
    return ensemble, results


def main():
    """Main training pipeline"""
    parser = argparse.ArgumentParser(description='Train bucket-based prediction engine')
    parser.add_argument('xes_file', type=str, help='Path to XES event log file')
    parser.add_argument('--output-dir', type=str, default='trained_models',
                       help='Output directory for trained models')
    parser.add_argument('--max-bucket', type=int, default=10,
                       help='Maximum bucket size')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Training epochs for bucket models')
    parser.add_argument('--phase1-only', action='store_true',
                       help='Train only Phase 1 (outcome prediction)')
    parser.add_argument('--phase2-only', action='store_true',
                       help='Train only Phase 2 (bucket models)')
    
    args = parser.parse_args()
    
    # Load and process event log
    print("="*70)
    print("BUCKET-BASED PREDICTIVE PROCESS MONITORING ENGINE")
    print("="*70)
    
    processor = EventLogProcessor(args.xes_file)
    processor.build_traces()
    
    # Build activity encoder
    activity_encoder = ActivityEncoder()
    activity_encoder.fit(processor.df)
    
    # Save encoder
    import joblib
    os.makedirs(args.output_dir, exist_ok=True)
    joblib.dump(activity_encoder, os.path.join(args.output_dir, 'activity_encoder.pkl'))
    
    # Train Phase 1
    if not args.phase2_only:
        outcome_predictor = train_phase1_outcome(
            processor, activity_encoder, args.output_dir
        )
    
    # Train Phase 2
    if not args.phase1_only:
        ensemble, results = train_phase2_buckets(
            processor, activity_encoder, args.output_dir,
            max_bucket=args.max_bucket, epochs=args.epochs
        )
    
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE!")
    print("="*70)
    print(f"\nModels saved to: {args.output_dir}")
    print("\nNext steps:")
    print("  1. Test predictions with API")
    print("  2. Integrate with visualization (Group 5)")
    print("  3. Deploy to production")


if __name__ == '__main__':
    main()
