"""
Main Training Script
Trains all models (Outcome Prediction + LSTM) and saves them
"""

import os
import sys
import argparse
from datetime import datetime

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_preprocessing import XESDataPreprocessor
from outcome_prediction import EnsembleOutcomePredictor
from lstm_models import CombinedLSTMPredictor


def train_all_models(xes_file_path: str, output_dir: str, 
                     outcome_only: bool = False, lstm_only: bool = False):
    """
    Train all prediction models
    
    Args:
        xes_file_path: Path to XES event log file
        output_dir: Directory to save trained models
        outcome_only: Train only outcome prediction models
        lstm_only: Train only LSTM models
    """
    print("\n" + "="*80)
    print("PROCESS MINING PREDICTION MODEL TRAINING")
    print("="*80)
    print(f"XES File: {xes_file_path}")
    print(f"Output Directory: {output_dir}")
    print(f"Training Mode: ", end="")
    if outcome_only:
        print("Outcome Prediction Only")
    elif lstm_only:
        print("LSTM Models Only")
    else:
        print("All Models")
    print("="*80 + "\n")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize preprocessor
    print("="*80)
    print("STEP 1: LOADING AND PREPROCESSING DATA")
    print("="*80)
    
    preprocessor = XESDataPreprocessor(xes_file_path)
    preprocessor.load_xes()
    preprocessor.convert_to_dataframe()
    preprocessor.clean_data()
    
    # Phase 1: Outcome Prediction
    if not lstm_only:
        print("\n" + "="*80)
        print("STEP 2: TRAINING OUTCOME PREDICTION MODELS (PHASE 1)")
        print("="*80)
        
        X_train, X_test, y_train, y_test, feature_columns = preprocessor.prepare_outcome_prediction_data()
        
        # Train ensemble
        ensemble = EnsembleOutcomePredictor()
        ensemble.train(X_train, y_train, feature_columns)
        
        # Evaluate
        ensemble.evaluate(X_test, y_test, preprocessor.outcome_encoder)
        
        # Save
        ensemble_dir = os.path.join(output_dir, 'ensemble')
        ensemble.save(ensemble_dir)
    
    # Phase 2: LSTM Models
    if not outcome_only:
        print("\n" + "="*80)
        print("STEP 3: TRAINING LSTM MODELS (PHASE 2)")
        print("="*80)
        
        lstm_data = preprocessor.prepare_sequence_data_for_lstm(max_case_length=50)
        
        # Train LSTM models
        lstm_predictor = CombinedLSTMPredictor(
            vocab_size=lstm_data['vocab_size'],
            max_length=lstm_data['max_length']
        )
        
        print("\n🔧 Model Improvements Active:")
        print("  ✓ Focal Loss (better for imbalanced data)")
        print("  ✓ Larger model (embedding_dim=128, lstm_units=256)")
        print("  ✓ Longer patience (20 epochs)")
        print("  ✓ More training epochs (50)")
        
        lstm_predictor.train(lstm_data, epochs=50, batch_size=64)
        
        # Evaluate
        lstm_predictor.evaluate(lstm_data)
        
        # Save
        lstm_dir = os.path.join(output_dir, 'lstm')
        lstm_predictor.save(lstm_dir)
    
    # Save preprocessor
    print("\n" + "="*80)
    print("STEP 4: SAVING PREPROCESSOR")
    print("="*80)
    
    preprocessor_path = os.path.join(output_dir, 'preprocessor.pkl')
    preprocessor.save_preprocessor(preprocessor_path)
    
    # Save training info
    info_path = os.path.join(output_dir, 'training_info.txt')
    with open(info_path, 'w') as f:
        f.write(f"Training completed at: {datetime.now()}\n")
        f.write(f"XES file: {xes_file_path}\n")
        f.write(f"Number of cases: {preprocessor.df['case:concept:name'].nunique()}\n")
        f.write(f"Number of events: {len(preprocessor.df)}\n")
        f.write(f"Unique activities: {preprocessor.df['concept:name'].nunique()}\n")
        if not lstm_only:
            f.write(f"Outcome classes: {len(preprocessor.outcome_encoder.classes_)}\n")
        if not outcome_only:
            f.write(f"Activity vocabulary size: {len(preprocessor.activity_encoder.classes_)}\n")
    
    print("\n" + "="*80)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"Models saved to: {output_dir}")
    print(f"Training info saved to: {info_path}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Train Process Mining Prediction Models')
    parser.add_argument('xes_file', type=str, help='Path to XES event log file')
    parser.add_argument('--output-dir', type=str, default='trained_models',
                       help='Directory to save trained models (default: trained_models)')
    parser.add_argument('--outcome-only', action='store_true',
                       help='Train only outcome prediction models')
    parser.add_argument('--lstm-only', action='store_true',
                       help='Train only LSTM models')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not os.path.exists(args.xes_file):
        print(f"Error: XES file not found: {args.xes_file}")
        sys.exit(1)
    
    if args.outcome_only and args.lstm_only:
        print("Error: Cannot specify both --outcome-only and --lstm-only")
        sys.exit(1)
    
    # Train models
    try:
        train_all_models(
            args.xes_file,
            args.output_dir,
            outcome_only=args.outcome_only,
            lstm_only=args.lstm_only
        )
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
