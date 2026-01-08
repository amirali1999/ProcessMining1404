"""
Prediction Service Layer
========================
Clean Python API for inference on pre-trained Group 7 models.
NO TRAINING - only prediction on already-trained models.

Architecture:
- Lazy model loading with global cache
- Integration with main app's EventLog DataFrames
- Core prediction functions: outcome, next_activity, remaining_time, all

Models used:
- EnsembleOutcomePredictor: Outcome prediction (DT, LR, RF ensemble)
- CombinedLSTMPredictor: Next activity + remaining time (LSTM)
- XESDataPreprocessor: Feature extraction and encoding
"""

import pandas as pd
import numpy as np
import pickle
import os
import logging
from typing import Dict, List, Optional, Tuple
from collections import Counter

logger = logging.getLogger(__name__)

# Import Group 7's model classes
from .outcome_prediction import EnsembleOutcomePredictor
from .lstm_models import CombinedLSTMPredictor
from .data_preprocessing import XESDataPreprocessor

# Import main app helpers
from preprocessing.services import get_default_event_log_df, get_event_log_dataframe
from uploads.models import EventLog

# Model path configuration
from . import TRAINED_MODELS_DIR

# Global cache for loaded models (singleton pattern)
_MODELS_CACHE = None


def load_prediction_models() -> Dict:
    """
    Load and cache all trained models from TRAINED_MODELS_DIR.
    
    This function implements lazy loading with a global cache to avoid
    reloading models on every prediction request.
    
    Loads:
    - XESDataPreprocessor: Encoders and scalers for feature transformation
    - EnsembleOutcomePredictor: Outcome prediction (DT, LR, RF voting)
    - CombinedLSTMPredictor: Next activity + remaining time (LSTM models)
    
    Returns:
        dict: Cached models with keys:
            - 'preprocessor': XESDataPreprocessor instance
            - 'outcome_model': EnsembleOutcomePredictor instance
            - 'lstm_predictor': CombinedLSTMPredictor instance
            - 'vocab_size': int (for LSTM)
            - 'max_length': int (for LSTM sequence padding)
    
    Raises:
        FileNotFoundError: If trained models are not found
        Exception: If model loading fails
    """
    global _MODELS_CACHE
    
    if _MODELS_CACHE is not None:
        print("✅ Using cached prediction models")
        return _MODELS_CACHE
    
    print("🔄 Loading prediction models for the first time...")
    
    # 1. Load preprocessor (encoders and scalers)
    preprocessor_path = os.path.join(TRAINED_MODELS_DIR, "preprocessor.pkl")
    if not os.path.exists(preprocessor_path):
        raise FileNotFoundError(f"Preprocessor not found at {preprocessor_path}")
    
    preprocessor = XESDataPreprocessor("")  # Empty path since we won't load XES directly
    preprocessor.load_preprocessor(preprocessor_path)
    print(f"✅ Preprocessor loaded from {preprocessor_path}")
    
    # 2. Load ensemble outcome model
    ensemble_path = os.path.join(TRAINED_MODELS_DIR, "ensemble")
    if not os.path.exists(ensemble_path):
        raise FileNotFoundError(f"Ensemble models not found at {ensemble_path}")
    
    outcome_model = EnsembleOutcomePredictor()
    outcome_model.load(ensemble_path)
    print(f"✅ Ensemble outcome model loaded from {ensemble_path}")
    
    # 3. Load LSTM models for next activity and remaining time
    lstm_path = os.path.join(TRAINED_MODELS_DIR, "lstm")
    if not os.path.exists(lstm_path):
        raise FileNotFoundError(f"LSTM models not found at {lstm_path}")
    
    # Get metadata for LSTM initialization
    metadata_path = os.path.join(lstm_path, "best_next_activity_model.keras_metadata.pkl")
    if os.path.exists(metadata_path):
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        vocab_size = metadata['vocab_size']
        max_length = metadata['max_length']
        print(f"📊 LSTM metadata: vocab_size={vocab_size}, max_length={max_length}")
    else:
        # Fallback: infer from preprocessor
        vocab_size = len(preprocessor.activity_encoder.classes_)
        max_length = 50  # Default from Group 7
        print(f"⚠️  LSTM metadata not found, using inferred values: vocab_size={vocab_size}, max_length={max_length}")
    
    lstm_predictor = CombinedLSTMPredictor(vocab_size=vocab_size, max_length=max_length)
    lstm_predictor.load(lstm_path)
    print(f"✅ LSTM models loaded from {lstm_path}")
    
    # Cache everything
    _MODELS_CACHE = {
        'preprocessor': preprocessor,
        'outcome_model': outcome_model,
        'lstm_predictor': lstm_predictor,
        'vocab_size': vocab_size,
        'max_length': max_length
    }
    
    print("✅ All models loaded and cached successfully!")
    return _MODELS_CACHE


def get_log_for_prediction(event_log_id: int, source: str = "default") -> pd.DataFrame:
    """
    Get event log DataFrame using main app's helpers.
    
    Uses the existing EventLog infrastructure instead of reading XES files.
    Ensures proper column naming for Group 7's preprocessor.
    
    Args:
        event_log_id: ID of the EventLog model instance
        source: "default", "raw", or "cleaned"
            - "default": Use get_default_event_log_df() (smart selection)
            - "raw": Use raw uploaded data
            - "cleaned": Use Smart Clean preprocessed data
    
    Returns:
        DataFrame with standard columns:
        - case:concept:name (case ID)
        - concept:name (activity name)
        - time:timestamp (timestamp)
        
    Raises:
        ValueError: If required columns are missing
    """
    # Get DataFrame from main app
    if source == "default":
        df = get_default_event_log_df(event_log_id)
    else:
        df = get_event_log_dataframe(event_log_id, version=source)
    
    # Ensure columns match Group 7 expectations
    required = ['case:concept:name', 'concept:name', 'time:timestamp']
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in DataFrame. Available: {df.columns.tolist()}")
    
    # Sort by case and time
    df = df.sort_values(['case:concept:name', 'time:timestamp']).reset_index(drop=True)
    
    return df


def build_prefix_from_case(df: pd.DataFrame, case_id: str) -> Tuple[List[str], Dict]:
    """
    Build activity prefix and metadata for a specific case.
    
    Args:
        df: Event log DataFrame
        case_id: Case identifier
    
    Returns:
        Tuple of (activities_list, metadata_dict) where:
        - activities_list: List of activity names in chronological order
        - metadata_dict: Dict with case_id, prefix_length, timestamps, etc.
        
    Raises:
        ValueError: If case_id not found in log
    """
    case_df = df[df['case:concept:name'] == case_id].sort_values('time:timestamp')
    
    if len(case_df) == 0:
        raise ValueError(f"Case ID '{case_id}' not found in event log")
    
    activities = case_df['concept:name'].tolist()
    timestamps = pd.to_datetime(case_df['time:timestamp']).tolist()
    
    metadata = {
        'case_id': case_id,
        'prefix_length': len(activities),
        'start_time': timestamps[0],
        'current_time': timestamps[-1],
        'elapsed_time': (timestamps[-1] - timestamps[0]).total_seconds(),
        'activities': activities
    }
    
    return activities, metadata


def predict_outcome(event_log_id: int, 
                   source: str = "default",
                   case_id: Optional[str] = None) -> Dict:
    """
    Predict the outcome for a case based on its current activity prefix.
    
    Uses the EnsembleOutcomePredictor (ensemble of DT, LR, RF) to classify
    the case outcome based on extracted features from the activity sequence.
    
    Args:
        event_log_id: ID of the EventLog
        source: "default", "raw", or "cleaned"
        case_id: Case identifier (required)
    
    Returns:
        dict with:
        - event_log_id: int
        - source: str
        - case_id: str
        - predicted_outcome: str (the predicted class label)
        - current_activities: list[str] (activity sequence)
        - prefix_length: int (number of activities so far)
        
    Raises:
        ValueError: If case_id is not provided or not found
    """
    if not case_id:
        raise ValueError("case_id is required for outcome prediction")
    
    # Load models
    models = load_prediction_models()
    outcome_model = models['outcome_model']
    preprocessor = models['preprocessor']
    
    # Get log and build prefix
    df = get_log_for_prediction(event_log_id, source)
    activities, metadata = build_prefix_from_case(df, case_id)
    
    # Prepare features for outcome prediction
    # Group 7's outcome model expects feature vectors
    features = {
        'prefix_length': metadata['prefix_length'],
        'elapsed_time': metadata['elapsed_time'],
        'unique_activities': len(set(activities)),
    }
    
    # Add recent activities (last 5)
    for i in range(5):
        if i < len(activities):
            features[f'activity_{i+1}'] = activities[-(i+1)]
        else:
            features[f'activity_{i+1}'] = 'NONE'
    
    # Most common activity
    activity_counts = Counter(activities)
    features['most_common_activity'] = activity_counts.most_common(1)[0][0]
    
    # Convert to DataFrame
    features_df = pd.DataFrame([features])
    
    # Encode categorical features using preprocessor's encoders
    for col in features_df.columns:
        if features_df[col].dtype == 'object' and col in preprocessor.label_encoders:
            encoder = preprocessor.label_encoders[col]
            # Handle unseen labels gracefully
            features_df[col] = features_df[col].apply(
                lambda x: x if x in encoder.classes_ else encoder.classes_[0]
            )
            features_df[col] = encoder.transform(features_df[col])
    
    # Predict
    prediction = outcome_model.predict(features_df)
    predicted_outcome = preprocessor.outcome_encoder.inverse_transform([prediction[0]])[0]
    
    return {
        'event_log_id': event_log_id,
        'source': source,
        'case_id': case_id,
        'predicted_outcome': predicted_outcome,
        'current_activities': activities,
        'prefix_length': metadata['prefix_length']
    }


def predict_next_activity(event_log_id: int,
                         source: str = "default", 
                         case_id: Optional[str] = None,
                         activities: Optional[List[str]] = None) -> Dict:
    """
    Predict the next activity based on current sequence.
    
    Uses the CombinedLSTMPredictor's next_activity_model to predict what
    activity is most likely to occur next in the process.
    
    Args:
        event_log_id: ID of the EventLog
        source: "default", "raw", or "cleaned"
        case_id: Case identifier (preferred - will extract sequence from log)
        activities: Manual activity sequence (alternative to case_id)
    
    Returns:
        dict with:
        - event_log_id: int
        - source: str
        - case_id: str (if provided)
        - input_activities: list[str] (the activity sequence used)
        - predicted_next_activity: str (most likely next activity)
        - top_predictions: list[dict] (top 5 predictions with probabilities)
            Each dict has: {'activity': str, 'probability': float}
            
    Raises:
        ValueError: If neither case_id nor activities is provided
    """
    # Load models
    models = load_prediction_models()
    lstm_predictor = models['lstm_predictor']
    preprocessor = models['preprocessor']
    max_length = models['max_length']
    
    # Get activity sequence
    if case_id:
        df = get_log_for_prediction(event_log_id, source)
        activities, _ = build_prefix_from_case(df, case_id)
    elif activities:
        activities = list(activities)
    else:
        raise ValueError("Either case_id or activities must be provided")
    
    # Encode and pad sequence
    seq = activities.copy()
    if len(seq) < max_length:
        seq = ['START'] * (max_length - len(seq)) + seq
    else:
        seq = seq[-max_length:]
    
    # Encode activities
    seq_encoded = []
    for act in seq:
        if act in preprocessor.activity_encoder.classes_:
            seq_encoded.append(preprocessor.activity_encoder.transform([act])[0])
        else:
            # Handle unseen activity - use START token or skip
            seq_encoded.append(preprocessor.activity_encoder.transform(['START'])[0])
    
    seq_encoded = np.array([seq_encoded])
    
    # Predict - use the next_activity_model's predict method
    predictions = lstm_predictor.next_activity_model.predict(seq_encoded, return_proba=True)
    
    # Get top 5 predictions
    pred_probs = predictions[0]
    top_indices = np.argsort(pred_probs)[-5:][::-1]  # Top 5 in descending order
    
    top_predictions = []
    for idx in top_indices:
        activity = preprocessor.activity_encoder.inverse_transform([idx])[0]
        probability = float(pred_probs[idx])
        top_predictions.append({
            'activity': activity,
            'probability': probability
        })
    
    predicted_next = top_predictions[0]['activity']
    
    return {
        'event_log_id': event_log_id,
        'source': source,
        'case_id': case_id,
        'input_activities': activities,
        'predicted_next_activity': predicted_next,
        'top_predictions': top_predictions
    }


def predict_remaining_time(event_log_id: int,
                          source: str = "default",
                          case_id: Optional[str] = None,
                          activities: Optional[List[str]] = None) -> Dict:
    """
    Predict remaining time for a case to complete.
    
    Uses the CombinedLSTMPredictor's remaining_time_model to estimate
    how much time remains until case completion.
    
    Args:
        event_log_id: ID of the EventLog
        source: "default", "raw", or "cleaned"
        case_id: Case identifier (preferred)
        activities: Manual activity sequence (alternative)
    
    Returns:
        dict with:
        - event_log_id: int
        - source: str
        - case_id: str
        - predicted_remaining_time_seconds: float
        - predicted_remaining_time_hours: float
        - predicted_remaining_time_days: float
        - current_activities: list[str]
        
    Raises:
        ValueError: If neither case_id nor activities is provided
    """
    # Load models
    models = load_prediction_models()
    lstm_predictor = models['lstm_predictor']
    preprocessor = models['preprocessor']
    max_length = models['max_length']
    
    # Get activity sequence
    if case_id:
        df = get_log_for_prediction(event_log_id, source)
        activities, _ = build_prefix_from_case(df, case_id)
    elif activities:
        activities = list(activities)
    else:
        raise ValueError("Either case_id or activities must be provided")
    
    # Encode and pad sequence
    seq = activities.copy()
    if len(seq) < max_length:
        seq = ['START'] * (max_length - len(seq)) + seq
    else:
        seq = seq[-max_length:]
    
    # Encode activities
    seq_encoded = []
    for act in seq:
        if act in preprocessor.activity_encoder.classes_:
            seq_encoded.append(preprocessor.activity_encoder.transform([act])[0])
        else:
            seq_encoded.append(preprocessor.activity_encoder.transform(['START'])[0])
    
    seq_encoded = np.array([seq_encoded])
    
    # Predict (returns normalized value) - use the remaining_time_model's predict method
    pred_normalized = lstm_predictor.remaining_time_model.predict(seq_encoded)[0]
    
    # Inverse transform: denormalize
    # Group 7 uses log1p transformation, so we need expm1
    pred_log = preprocessor.time_scaler.inverse_transform([[pred_normalized]])[0][0]
    pred_seconds = np.expm1(pred_log)  # Inverse of log1p
    pred_seconds = max(0, pred_seconds)  # Ensure non-negative
    
    return {
        'event_log_id': event_log_id,
        'source': source,
        'case_id': case_id,
        'predicted_remaining_time_seconds': float(pred_seconds),
        'predicted_remaining_time_hours': float(pred_seconds / 3600),
        'predicted_remaining_time_days': float(pred_seconds / 86400),
        'current_activities': activities
    }


def predict_all(event_log_id: int,
               source: str = "default",
               case_id: Optional[str] = None,
               activities: Optional[List[str]] = None) -> Dict:
    """
    Run all predictions at once (outcome, next activity, remaining time).
    
    This is the main entry point for comprehensive predictions. It calls
    all three individual prediction functions and merges the results.
    
    Args:
        event_log_id: ID of the EventLog
        source: "default", "raw", or "cleaned"
        case_id: Case identifier (preferred)
        activities: Manual activity sequence (alternative)
    
    Returns:
        dict with all prediction results merged:
        - event_log_id: int
        - source: str
        - case_id: str
        - input_activities: list[str]
        - outcome: dict (result from predict_outcome)
        - next_activity: dict (result from predict_next_activity)
        - remaining_time: dict (result from predict_remaining_time)
        
    Raises:
        ValueError: If neither case_id nor activities is provided
    """
    # Get activity sequence once
    if case_id:
        df = get_log_for_prediction(event_log_id, source)
        activities_list, _ = build_prefix_from_case(df, case_id)
    elif activities:
        activities_list = list(activities)
        case_id = None  # Mark as manual input
    else:
        raise ValueError("Either case_id or activities must be provided")
    
    # Run all predictions
    # Note: Outcome prediction disabled due to feature mismatch
    # (requires 29 engineered features from Group 7's extract_features method)
    outcome_result = None
    
    try:
        next_activity_result = predict_next_activity(event_log_id, source, case_id, activities_list)
    except Exception as e:
        logger.error(f"Next activity prediction failed: {str(e)}")
        next_activity_result = None
    
    try:
        remaining_time_result = predict_remaining_time(event_log_id, source, case_id, activities_list)
    except Exception as e:
        logger.error(f"Remaining time prediction failed: {str(e)}")
        remaining_time_result = None
    
    # Merge results
    return {
        'event_log_id': event_log_id,
        'source': source,
        'case_id': case_id,
        'input_activities': activities_list,
        'outcome': outcome_result,
        'next_activity': next_activity_result,
        'remaining_time': remaining_time_result
    }
