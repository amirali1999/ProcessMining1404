
import os
import sys
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
import traceback
from collections import Counter

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from lstm_models import CombinedLSTMPredictor
from data_preprocessing import XESDataPreprocessor
from outcome_prediction import EnsembleOutcomePredictor

# Global variables
preprocessor = None

def load_models():
    print("="*80)
    print("LOADING MODELS")
    print("="*80)
    
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(project_root, 'trained_models', 'lstm')
    outcome_dir = os.path.join(project_root, 'trained_models', 'ensemble')
    preprocessor_path = os.path.join(project_root, 'trained_models', 'preprocessor.pkl')
    
    # 1. Load Preprocessor
    global preprocessor
    if os.path.exists(preprocessor_path):
        preprocessor = XESDataPreprocessor('')
        preprocessor.load_preprocessor(preprocessor_path)
        print("✓ Preprocessor loaded")
    else:
        print("Error: Preprocessor not found")
        return None, None

    # 2. Load LSTM Models
    metadata_path = os.path.join(models_dir, 'best_next_activity_model.keras_metadata.pkl')
    if not os.path.exists(metadata_path):
        metadata_path = os.path.join(models_dir, 'next_activity_lstm_metadata.pkl')
        
    if not os.path.exists(metadata_path):
        print("Warning: Metadata not found, inferring from preprocessor...")
        vocab_size = len(preprocessor.activity_encoder.classes_)
        metadata = {'vocab_size': vocab_size, 'max_length': 50}
    else:
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
            
    print(f"Metadata: vocab_size={metadata['vocab_size']}, max_length={metadata['max_length']}")
    
    predictor = CombinedLSTMPredictor(
        vocab_size=metadata['vocab_size'],
        max_length=metadata['max_length']
    )
    
    try:
        predictor.load(models_dir)
        print("✓ LSTM models loaded")
    except Exception as e:
        print(f"Error loading LSTM models: {e}")
        return None, None
    
    # 3. Load Outcome Model
    outcome_model = None
    if os.path.exists(outcome_dir):
        outcome_model = EnsembleOutcomePredictor()
        outcome_model.load(outcome_dir)
        print("✓ Outcome model loaded")
    else:
        print("Warning: Outcome model not found")
        
    return predictor, outcome_model

def prepare_sequence(activities, max_length):
    if preprocessor is None:
        return None
        
    indices = []
    for act in activities:
        # Handle unknown activities
        if act in preprocessor.activity_encoder.classes_:
            idx = preprocessor.activity_encoder.transform([act])[0]
            # Ensure index is within vocab range (1-based usually)
            indices.append(idx)
        else:
            print(f"  [Warn] Unknown activity: {act}")
            indices.append(0) # 0 is usually reserved for padding or unknown
            
    # Pad or truncate
    if len(indices) > max_length:
        indices = indices[-max_length:]
    else:
        indices = [0] * (max_length - len(indices)) + indices
        
    return np.array([indices])

def prepare_features(activities):
    if preprocessor is None:
        return None
        
    features = {}
    features['prefix_length'] = len(activities)
    features['elapsed_time'] = 0 # Dummy
    
    # Activity-based features
    for i in range(min(5, len(activities))):
        features[f'activity_{i+1}'] = activities[-(i+1)]
    for i in range(len(activities), 5):
        features[f'activity_{i+1}'] = 'NONE'
        
    features['unique_activities'] = len(set(activities))
    if activities:
        features['most_common_activity'] = Counter(activities).most_common(1)[0][0]
    else:
        features['most_common_activity'] = 'NONE'
        
    # Encode
    features_df = pd.DataFrame([features])
    
    # Debug: Print columns before encoding
    # print(f"Columns before encoding: {features_df.columns.tolist()}")
    
    # Simple encoding simulation based on api_views logic
    for col in features_df.columns:
        if col in preprocessor.label_encoders:
            encoder = preprocessor.label_encoders[col]
            try:
                features_df[col] = encoder.transform(features_df[col].astype(str))
            except:
                features_df[col] = encoder.transform([encoder.classes_[0]])[0]
        elif col.startswith('activity_') or col == 'most_common_activity':
            # Handle activity columns and most_common_activity using activity_encoder
            try:
                val = str(features_df[col].iloc[0])
                if val in preprocessor.activity_encoder.classes_:
                    features_df[col] = preprocessor.activity_encoder.transform([val])
                else:
                    features_df[col] = 0
            except:
                features_df[col] = 0
                
    return features_df

def run_test_case(name, activities, predictor, outcome_model):
    print(f"\n{'='*20} Test Case: {name} {'='*20}")
    print(f"Input Sequence: {activities}")
    
    if not activities and name != "Empty Sequence":
        print("Skipping empty sequence check for non-empty test case")
        
    # 1. LSTM Prediction
    # Use max_length from one of the sub-models
    max_length = predictor.next_activity_model.max_length
    seq = prepare_sequence(activities, max_length)
    if seq is not None:
        try:
            preds = predictor.predict(seq)
            
            # Decode Next Activity
            next_act_idx = preds['next_activity'][0]
            if next_act_idx < len(preprocessor.activity_encoder.classes_):
                next_act = preprocessor.activity_encoder.inverse_transform([next_act_idx])[0]
            else:
                next_act = f"Index {next_act_idx} (Out of bounds)"
                
            print(f"✓ Next Activity: {next_act}")
            
            # Decode Time
            norm_time = preds['remaining_time'][0]
            if hasattr(preprocessor, 'time_scaler'):
                log_time = preprocessor.time_scaler.inverse_transform([[norm_time]])[0][0]
                real_time = np.expm1(log_time)
                print(f"✓ Remaining Time: {real_time/86400:.2f} days")
            else:
                print(f"✓ Remaining Time (Norm): {norm_time:.4f}")
                
        except Exception as e:
            print(f"✗ LSTM Prediction Failed: {e}")
            # traceback.print_exc()
    
    # 2. Outcome Prediction
    if outcome_model:
        try:
            features = prepare_features(activities)
            
            # Align columns if possible (simplified)
            rf_model = outcome_model.models.get('random_forest')
            if rf_model and hasattr(rf_model, 'feature_columns'):
                # Add missing columns with 0
                for col in rf_model.feature_columns:
                    if col not in features.columns:
                        features[col] = 0
                features = features[rf_model.feature_columns]
            
            pred_idx = outcome_model.predict(features, method='best')[0]
            
            # Decode Outcome
            outcome_name = "Unknown"
            if hasattr(preprocessor.outcome_encoder, 'classes_'):
                if pred_idx < len(preprocessor.outcome_encoder.classes_):
                    outcome_name = preprocessor.outcome_encoder.inverse_transform([pred_idx])[0]
                else:
                    outcome_name = f"Index {pred_idx}"
            
            print(f"✓ Raw Outcome: {outcome_name} (Index {pred_idx})")
            
            # Test Mapping Logic
            fallback_classes = ["A", "B", "C", "E", "H", "O", "UNKNOWN", "W"]
            mapped_outcome = outcome_name
            
            # Apply API logic
            if outcome_name.upper() == "UNKNOWN":
                mapped_outcome = "Standard Processing"
            elif outcome_name.startswith("Index"):
                 if 0 <= pred_idx < len(fallback_classes):
                    mapped_outcome = fallback_classes[pred_idx]
                    if mapped_outcome == "UNKNOWN":
                        mapped_outcome = "Standard Processing"
            
            print(f"✓ Mapped Outcome: {mapped_outcome}")
            
        except Exception as e:
            print(f"✗ Outcome Prediction Failed: {e}")
            # traceback.print_exc()

def test_best_models():
    predictor, outcome_model = load_models()
    if not predictor:
        return

    test_cases = [
        ("Happy Path", ['NEW', 'CODE OK', 'MANUAL']),
        ("Short Sequence", ['NEW']),
        ("Unknown Activity", ['NEW', 'SUPER_WEIRD_ACTIVITY', 'MANUAL']),
        ("Empty Sequence", []),
        ("Long Sequence", ['NEW', 'CODE OK', 'MANUAL', 'BILLED', 'STORNO', 'REJECTED', 'NEW', 'CODE OK']),
        ("Outcome Mapping Check", ['NEW', 'CODE OK', 'MANUAL']) 
    ]
    
    for name, acts in test_cases:
        run_test_case(name, acts, predictor, outcome_model)

if __name__ == "__main__":
    test_best_models()

