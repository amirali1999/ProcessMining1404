"""
Quick Test Script
Test the bucket prediction engine on a subset of data
"""

import sys
import os

# Add to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader import EventLogProcessor, PrefixBucketer, ActivityEncoder
from outcome_models import OutcomePredictor
from bucket_models import BucketEnsemble
from sklearn.model_selection import train_test_split
import numpy as np


def quick_test(xes_file, max_cases=5000):
    """
    Quick test on subset of data
    
    Args:
        xes_file: Path to XES file
        max_cases: Maximum number of cases to use
    """
    print("="*70)
    print("QUICK TEST - BUCKET PREDICTION ENGINE")
    print("="*70)
    
    # Load data
    print(f"\n📂 Loading {max_cases} cases from {xes_file}...")
    processor = EventLogProcessor(xes_file)
    processor.build_traces()
    
    # Limit to max_cases
    if len(processor.traces) > max_cases:
        case_ids = list(processor.traces.keys())[:max_cases]
        processor.traces = {cid: processor.traces[cid] for cid in case_ids}
        processor.case_outcome = {cid: processor.case_outcome[cid] for cid in case_ids}
        processor.case_timestamps = {cid: processor.case_timestamps[cid] for cid in case_ids}
        processor.df = processor.df[processor.df['case_id'].isin(case_ids)]
        print(f"   Limited to {len(processor.traces)} cases")
    
    # Build encoder
    activity_encoder = ActivityEncoder()
    activity_encoder.fit(processor.df)
    
    # Test Phase 1: Outcome Prediction
    print("\n" + "="*70)
    print("PHASE 1 TEST: OUTCOME PREDICTION")
    print("="*70)
    
    prefix_samples = processor.generate_prefixes()
    outcome_predictor = OutcomePredictor(activity_encoder)
    X, y = outcome_predictor.prepare_data(prefix_samples, max_prefix_length=5)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42, stratify=y_train
    )
    
    # Train
    outcome_predictor.train(X_train, y_train, X_val, y_val)
    
    # Evaluate
    print("\n📊 Test Set Evaluation:")
    outcome_predictor.evaluate(X_test, y_test)
    
    # Test Phase 2: Bucket Models
    print("\n" + "="*70)
    print("PHASE 2 TEST: BUCKET MODELS")
    print("="*70)
    
    # Bucket prefixes
    bucketer = PrefixBucketer(max_bucket=10)
    buckets = bucketer.bucket_prefixes(prefix_samples)
    
    # Initialize ensemble
    ensemble = BucketEnsemble(vocab_size=activity_encoder.vocab_size, max_bucket=10)
    bucket_data = ensemble.prepare_bucket_data(buckets, activity_encoder)
    
    # Train
    ensemble.train_all_buckets(bucket_data, epochs=5, batch_size=64)
    
    # Evaluate
    print("\n📊 Bucket Model Evaluation:")
    results = ensemble.evaluate_all_buckets(bucket_data)
    
    # Summary
    print("\n" + "="*70)
    print("QUICK TEST COMPLETE!")
    print("="*70)
    
    print("\n📈 Summary:")
    print(f"   Total cases: {len(processor.traces)}")
    print(f"   Total prefixes: {len(prefix_samples)}")
    print(f"   Buckets created: {len(buckets)}")
    print(f"   Vocabulary size: {activity_encoder.vocab_size}")
    
    # Test prediction
    print("\n🔮 Testing prediction on sample case...")
    sample_case_id = list(processor.traces.keys())[0]
    sample_prefix = processor.traces[sample_case_id][:3]
    
    print(f"   Case ID: {sample_case_id}")
    print(f"   Prefix: {sample_prefix}")
    
    # Predict outcome
    prefix_sample = {'prefix_activities': sample_prefix, 'prefix_length': len(sample_prefix)}
    outcome, confidence = outcome_predictor.predict(prefix_sample)
    print(f"\n   Predicted Outcome: {outcome} (confidence: {confidence:.4f})")
    
    # Predict next activity
    encoded_prefix = activity_encoder.encode_prefix(sample_prefix)
    prediction = ensemble.predict(encoded_prefix, activity_encoder)
    print(f"   Predicted Next Activity: {prediction['predicted_next_activity']}")
    print(f"   Top-3 candidates:")
    for i, cand in enumerate(prediction['top_k_candidates'][:3], 1):
        print(f"      {i}. {cand['activity']}: {cand['prob']:.4f}")
    print(f"   Predicted Remaining Time: {prediction['predicted_remaining_time']:.2f} hours")
    print(f"   Using Bucket: {prediction['bucket_id']}")
    
    print("\n✅ Test complete!")
    print("\nNext steps:")
    print("  1. Train on full dataset: python train_models.py <xes_file>")
    print("  2. Test API endpoints")
    print("  3. Integrate with visualization")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python quick_test.py <xes_file> [max_cases]")
        sys.exit(1)
    
    xes_file = sys.argv[1]
    max_cases = int(sys.argv[2]) if len(sys.argv) > 2 else 5000
    
    quick_test(xes_file, max_cases)
