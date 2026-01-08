"""
Django REST API for Process Mining Predictions
Provides endpoints for outcome, next activity, and remaining time predictions
"""

from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
import json
import numpy as np
import pandas as pd
import pm4py
import os
import pickle

# Import our models
from .outcome_prediction import OutcomePredictionModel, EnsembleOutcomePredictor
from .lstm_models import CombinedLSTMPredictor
from .data_preprocessing import XESDataPreprocessor

# Global variables for loaded models
# Check both project root and local app directory for trained_models
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCAL_MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'trained_models')
ROOT_MODELS_DIR = os.path.join(BASE_DIR, 'trained_models')

outcome_model = None
lstm_predictor = None
preprocessor = None
event_log_df = None


def demo_page(request):
    """Serve the demo page"""
    return render(request, 'demo.html')

def load_models():
    """Load all trained models"""
    global outcome_model, lstm_predictor, preprocessor, event_log_df
    
    print("Loading models...")
    
    # 1. Load preprocessor
    # Check root first (most likely location for shared preprocessor), then local
    preprocessor_path = os.path.join(ROOT_MODELS_DIR, 'preprocessor.pkl')
    if not os.path.exists(preprocessor_path):
        preprocessor_path = os.path.join(LOCAL_MODELS_DIR, 'preprocessor.pkl')
        
    if os.path.exists(preprocessor_path):
        preprocessor = XESDataPreprocessor('')
        preprocessor.load_preprocessor(preprocessor_path)
        print(f"Preprocessor loaded from {preprocessor_path}")
        
        # Check if outcome_encoder is fitted
        if hasattr(preprocessor.outcome_encoder, 'classes_'):
            print(f"Outcome encoder is fitted with {len(preprocessor.outcome_encoder.classes_)} classes")
        else:
            print("Warning: Outcome encoder is NOT fitted")
    else:
        print("Error: Preprocessor not found in root or local trained_models")
    
    # 2. Load outcome model (Ensemble)
    # Check local first (app specific), then root
    outcome_model_path = os.path.join(LOCAL_MODELS_DIR, 'ensemble')
    if not os.path.exists(outcome_model_path):
        outcome_model_path = os.path.join(ROOT_MODELS_DIR, 'ensemble')
        
    if os.path.exists(outcome_model_path):
        outcome_model = EnsembleOutcomePredictor()
        outcome_model.load(outcome_model_path)
        print(f"Outcome model loaded from {outcome_model_path}")
    else:
        print("Warning: Outcome model (ensemble) not found")
    
    # 3. Load LSTM models
    # Check root first (seems to be where they are), then local
    lstm_model_path = os.path.join(ROOT_MODELS_DIR, 'lstm')
    if not os.path.exists(lstm_model_path) or not os.listdir(lstm_model_path):
        lstm_model_path = os.path.join(LOCAL_MODELS_DIR, 'lstm')
        
    if os.path.exists(lstm_model_path):
        # Get vocab size and max length from metadata
        # Try new naming convention first
        metadata_path = os.path.join(lstm_model_path, 'best_next_activity_model.keras_metadata.pkl')
        if not os.path.exists(metadata_path):
            # Fallback to old naming
            metadata_path = os.path.join(lstm_model_path, 'next_activity_lstm_metadata.pkl')
            
        if os.path.exists(metadata_path):
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
            
            lstm_predictor = CombinedLSTMPredictor(
                vocab_size=metadata['vocab_size'],
                max_length=metadata['max_length']
            )
            lstm_predictor.load(lstm_model_path)
            print(f"LSTM models loaded from {lstm_model_path}")
        elif preprocessor is not None:
            # Fallback: Infer from preprocessor
            print("Metadata not found, inferring from preprocessor...")
            vocab_size = len(preprocessor.activity_encoder.classes_)
            lstm_predictor = CombinedLSTMPredictor(
                vocab_size=vocab_size,
                max_length=50  # Default
            )
            lstm_predictor.load(lstm_model_path)
            print(f"LSTM models loaded (inferred metadata) from {lstm_model_path}")
    else:
        print("Warning: LSTM models not found")
    
    # 4. Load event log for case lookup
    # Use BASE_DIR to find the event log folder
    xes_file = os.path.join(BASE_DIR, 'HospitalBilling-EventLog_1_all', 
                            'HospitalBilling-EventLog.xes')
    if os.path.exists(xes_file):
        log = pm4py.read_xes(xes_file)
        event_log_df = pm4py.convert_to_dataframe(log)
        print(f"Event log loaded: {len(event_log_df)} events")
    else:
        print(f"Warning: Event log not found at {xes_file}")
    
    print("All models loaded successfully")


def get_case_data(case_id: str):
    """
    Get data for a specific case
    
    Args:
        case_id: Case identifier
        
    Returns:
        Dictionary with case data
    """
    if event_log_df is None:
        return None
    
    case_data = event_log_df[event_log_df['case:concept:name'] == case_id]
    
    if len(case_data) == 0:
        return None
    
    case_data = case_data.sort_values('time:timestamp')
    
    return {
        'case_id': case_id,
        'activities': case_data['concept:name'].tolist(),
        'timestamps': case_data['time:timestamp'].tolist(),
        'attributes': case_data.iloc[0].to_dict()
    }


def prepare_case_features(case_data: dict):
    """
    Prepare features for a case
    
    Args:
        case_data: Case data dictionary
        
    Returns:
        Features ready for prediction
    """
    activities = case_data.get('activities', [])
    
    # Create feature dictionary similar to training
    features = {}
    features['prefix_length'] = len(activities)
    
    # Calculate elapsed time
    timestamps = case_data.get('timestamps', [])
    if timestamps and len(timestamps) > 0:
        elapsed_time = (timestamps[-1] - timestamps[0]).total_seconds()
    else:
        elapsed_time = 0 # Default if no timestamps
        
    features['elapsed_time'] = elapsed_time
    
    # Activity-based features
    for i in range(min(5, len(activities))):
        if i < len(activities):
            features[f'activity_{i+1}'] = activities[-(i+1)]
        else:
            features[f'activity_{i+1}'] = 'NONE'
    
    # Activity statistics
    features['unique_activities'] = len(set(activities))
    
    if activities:
        from collections import Counter
        activity_counts = Counter(activities)
        features['most_common_activity'] = activity_counts.most_common(1)[0][0]
    else:
        features['most_common_activity'] = 'NONE'
    
    # Case attributes
    attributes = case_data.get('attributes', {})
    for key, value in attributes.items():
        if key.startswith('case:') and key != 'case:concept:name':
            features[key] = value
    
    return features


def encode_features(features: dict):
    """
    Encode features using the preprocessor's encoders
    
    Args:
        features: Feature dictionary
        
    Returns:
        Encoded feature array
    """
    if preprocessor is None:
        raise ValueError("Preprocessor not loaded")
    
    # Create DataFrame with single row
    features_df = pd.DataFrame([features])
    
    # Encode categorical features
    for col in features_df.columns:
        if col in preprocessor.label_encoders and features_df[col].dtype == 'object':
            encoder = preprocessor.label_encoders[col]
            try:
                features_df[col] = encoder.transform(features_df[col].astype(str))
            except ValueError:
                # Unknown value - use most common class
                features_df[col] = encoder.transform([encoder.classes_[0]])[0]
        
        # Fallback for activity columns if specific encoder is missing
        elif col.startswith('activity_') and features_df[col].dtype == 'object':
            try:
                # Use the general activity encoder
                val = str(features_df[col].iloc[0])
                
                # Check if value is in encoder classes
                if val in preprocessor.activity_encoder.classes_:
                    features_df[col] = preprocessor.activity_encoder.transform([val])
                else:
                    # Handle 'NONE' or unknown values
                    # If 'NONE' is not in classes, use 0 or a default
                    features_df[col] = 0
            except Exception:
                features_df[col] = 0
                
        # Fallback for other object columns that couldn't be encoded
        elif features_df[col].dtype == 'object':
            try:
                # Try to convert to numeric if possible
                features_df[col] = pd.to_numeric(features_df[col])
            except:
                # If still object/string, set to 0 to avoid model crash
                features_df[col] = 0
    
    return features_df


def prepare_sequence(activities, max_length=50):
    """
    Prepare activity sequence for LSTM
    
    Args:
        activities: List of activities
        max_length: Maximum sequence length
        
    Returns:
        Encoded sequence
    """
    if preprocessor is None:
        raise ValueError("Preprocessor not loaded")
    
    # Encode activities
    encoded_activities = []
    for act in activities:
        try:
            encoded = preprocessor.activity_encoder.transform([act])[0]
        except ValueError:
            # Unknown activity - use 0
            encoded = 0
        encoded_activities.append(encoded)
    
    # Pad or truncate
    if len(encoded_activities) < max_length:
        # Pad with 0 (assuming 0 is padding)
        encoded_activities = [0] * (max_length - len(encoded_activities)) + encoded_activities
    else:
        encoded_activities = encoded_activities[-max_length:]
    
    return np.array([encoded_activities])


@csrf_exempt
@require_http_methods(["POST"])
def predict_outcome(request):
    """
    Predict outcome for a case
    
    POST /api/predict/outcome/
    Body: {"case_id": "case_123"}
    """
    try:
        data = json.loads(request.body)
        case_id = data.get('case_id')
        
        if not case_id:
            return JsonResponse({'error': 'case_id is required'}, status=400)
        
        # Get case data
        case_data = get_case_data(case_id)
        if case_data is None:
            return JsonResponse({'error': f'Case {case_id} not found'}, status=404)
        
        # Prepare features
        features = prepare_case_features(case_data)
        features_encoded = encode_features(features)
        
        # Make prediction
        if outcome_model is None:
            return JsonResponse({'error': 'Outcome model not loaded'}, status=500)
        
        prediction = outcome_model.predict(features_encoded, method='best')[0]
        
        if hasattr(preprocessor.outcome_encoder, 'classes_'):
            outcome_name = preprocessor.outcome_encoder.inverse_transform([prediction])[0]
        else:
            outcome_name = f"Class {prediction} (Label missing)"
        
        response = {
            'case_id': case_id,
            'predicted_outcome': outcome_name,
            'current_activities': case_data['activities'],
            'prefix_length': len(case_data['activities'])
        }
        
        return JsonResponse(response)
        
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


@csrf_exempt
@require_http_methods(["POST"])
def predict_next_activity(request):
    """
    Predict next activity for a case
    
    POST /api/predict/next-activity/
    Body: {"case_id": "case_123"}
    """
    try:
        data = json.loads(request.body)
        case_id = data.get('case_id')
        
        if not case_id:
            return JsonResponse({'error': 'case_id is required'}, status=400)
        
        # Get case data
        case_data = get_case_data(case_id)
        if case_data is None:
            return JsonResponse({'error': f'Case {case_id} not found'}, status=404)
        
        # Prepare sequence
        sequence = prepare_sequence(case_data['activities'])
        
        # Make prediction
        if lstm_predictor is None:
            return JsonResponse({'error': 'LSTM model not loaded'}, status=500)
        
        next_activity_encoded = lstm_predictor.next_activity_model.predict(sequence)[0]
        next_activity = preprocessor.activity_encoder.inverse_transform([next_activity_encoded])[0]
        
        # Get top-k predictions
        proba = lstm_predictor.next_activity_model.predict(sequence, return_proba=True)[0]
        top_k_indices = np.argsort(proba)[-5:][::-1]
        top_k_activities = preprocessor.activity_encoder.inverse_transform(top_k_indices)
        top_k_proba = proba[top_k_indices]
        
        response = {
            'case_id': case_id,
            'predicted_next_activity': next_activity,
            'current_activities': case_data['activities'],
            'top_predictions': [
                {'activity': act, 'probability': float(prob)}
                for act, prob in zip(top_k_activities, top_k_proba)
            ]
        }
        
        return JsonResponse(response)
        
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


@csrf_exempt
@require_http_methods(["POST"])
def predict_remaining_time(request):
    """
    Predict remaining time for a case
    
    POST /api/predict/remaining-time/
    Body: {"case_id": "case_123"}
    """
    try:
        data = json.loads(request.body)
        case_id = data.get('case_id')
        
        if not case_id:
            return JsonResponse({'error': 'case_id is required'}, status=400)
        
        # Get case data
        case_data = get_case_data(case_id)
        if case_data is None:
            return JsonResponse({'error': f'Case {case_id} not found'}, status=404)
        
        # Prepare sequence
        sequence = prepare_sequence(case_data['activities'])
        
        # Make prediction
        if lstm_predictor is None:
            return JsonResponse({'error': 'LSTM model not loaded'}, status=500)
        
        remaining_time_norm = lstm_predictor.remaining_time_model.predict(sequence)[0]
        
        # Denormalize time
        if hasattr(preprocessor, 'time_scaler'):
            log_time = preprocessor.time_scaler.inverse_transform([[remaining_time_norm]])[0][0]
            remaining_time_seconds = np.expm1(log_time)
        else:
            remaining_time_seconds = remaining_time_norm
        
        # Calculate elapsed time
        timestamps = case_data['time:timestamp'].tolist()
        elapsed_time = (timestamps[-1] - timestamps[0]).total_seconds()
        
        response = {
            'case_id': case_id,
            'predicted_remaining_time_seconds': float(remaining_time_seconds),
            'predicted_remaining_time_minutes': float(remaining_time_seconds / 60),
            'predicted_remaining_time_hours': float(remaining_time_seconds / 3600),
            'predicted_remaining_time_days': float(remaining_time_seconds / 86400),
            'elapsed_time_seconds': elapsed_time,
            'current_activities': case_data['concept:name'].tolist()
        }
        
        return JsonResponse(response)
        
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


@csrf_exempt
@require_http_methods(["POST"])
def predict_all(request):
    """
    Get all predictions for a case
    
    POST /api/predict/all/
    Body: {"case_id": "case_123"} OR {"activities": ["A", "B", "C"]}
    """
    try:
        data = json.loads(request.body)
        case_id = data.get('case_id')
        activities = data.get('activities')
        
        if not case_id and not activities:
            return JsonResponse({'error': 'Either case_id or activities list is required'}, status=400)
        
        # Get case data
        if case_id:
            case_data = get_case_data(case_id)
            if case_data is None:
                return JsonResponse({'error': f'Case {case_id} not found'}, status=404)
        else:
            # Construct case data from activities
            if isinstance(activities, str):
                activities = [a.strip() for a in activities.split(',')]
            
            case_data = {
                'activities': activities,
                'timestamps': [], # Dummy
                'attributes': {}  # Dummy
            }
        
        # Prepare features for outcome
        # Note: This might be less accurate without timestamps/attributes
        try:
            features = prepare_case_features(case_data)
            features_encoded = encode_features(features)
            
            # Align features with model expectations
            if outcome_model is not None:
                # Get feature columns from one of the sub-models (e.g., random_forest)
                # Assuming all models in ensemble use same features
                rf_model = outcome_model.models.get('random_forest')
                if rf_model and rf_model.feature_columns:
                    expected_cols = rf_model.feature_columns
                    # Reindex to ensure all columns exist and are in correct order
                    # Missing columns will be filled with NaN (which imputer will handle)
                    features_encoded = features_encoded.reindex(columns=expected_cols)
            
            outcome_ready = True
        except Exception as e:
            print(f"Could not prepare outcome features: {e}")
            outcome_ready = False
        
        # Prepare sequence for LSTM
        sequence = prepare_sequence(case_data['activities'])
        
        # Make predictions
        response = {
            'case_id': case_id if case_id else 'hypothetical',
            'current_activities': case_data['activities'],
            'prefix_length': len(case_data['activities']),
            'suggestion': {},
            'prediction': {}
        }
        
        # Outcome prediction
        if outcome_model is not None and outcome_ready:
            try:
                prediction = outcome_model.predict(features_encoded, method='best')[0]
                
                # Check if outcome_encoder is fitted
                if hasattr(preprocessor.outcome_encoder, 'classes_'):
                    outcome_name = preprocessor.outcome_encoder.inverse_transform([prediction])[0]
                    
                    # Map "UNKNOWN" to a more user-friendly term
                    if outcome_name.upper() == "UNKNOWN":
                        outcome_name = "Standard Processing"
                else:
                    # Fallback if encoder is not fitted (e.g. preprocessor from LSTM-only training)
                    # Manually map based on known sorted classes: A, B, C, E, H, O, UNKNOWN, W
                    fallback_classes = ["A", "B", "C", "E", "H", "O", "UNKNOWN", "W"]
                    if 0 <= prediction < len(fallback_classes):
                        outcome_name = fallback_classes[prediction]
                        if outcome_name.upper() == "UNKNOWN":
                            outcome_name = "Standard Processing"
                    else:
                        outcome_name = f"Class {prediction} (Label missing)"
                    
                response['prediction']['outcome'] = outcome_name
            except Exception as e:
                print(f"Outcome prediction error: {e}")
                response['prediction']['outcome_error'] = str(e)
                # Fallback if prediction fails
                response['prediction']['predicted_outcome'] = "Prediction Failed"
        else:
             # Fallback: Try to predict outcome using LSTM if available (some implementations do this)
             # Or just return a default/error if outcome model is missing
             if not outcome_ready:
                 response['prediction']['outcome_error'] = "Outcome features could not be prepared (missing context)"
             elif outcome_model is None:
                 response['prediction']['outcome_error'] = "Outcome model not loaded"
             
             # If we have next activity, we can sometimes infer outcome or just say "In Progress"
             if 'suggestion' in response and 'next_activity' in response['suggestion']:
                 # This is a heuristic fallback
                 response['prediction']['predicted_outcome'] = "In Progress (Model Unavailable)"
        
        # Next activity prediction
        if lstm_predictor is not None:
            try:
                next_activity_encoded = lstm_predictor.next_activity_model.predict(sequence)[0]
                next_activity = preprocessor.activity_encoder.inverse_transform([next_activity_encoded])[0]
                response['suggestion']['next_activity'] = next_activity
                
                # Remaining time prediction
                remaining_time_norm = lstm_predictor.remaining_time_model.predict(sequence)[0]
                
                # Denormalize time
                if hasattr(preprocessor, 'time_scaler'):
                    # Handle scalar output correctly
                    if isinstance(remaining_time_norm, (list, np.ndarray)):
                         val = remaining_time_norm[0] if len(remaining_time_norm) > 0 else 0
                    else:
                         val = remaining_time_norm
                         
                    log_time = preprocessor.time_scaler.inverse_transform([[val]])[0][0]
                    remaining_time_seconds = np.expm1(log_time)
                else:
                    remaining_time_seconds = remaining_time_norm
                    
                response['prediction']['remaining_time'] = {
                    'seconds': float(remaining_time_seconds),
                    'minutes': float(remaining_time_seconds / 60),
                    'hours': float(remaining_time_seconds / 3600),
                    'days': float(remaining_time_seconds / 86400)
                }
            except Exception as e:
                print(f"LSTM prediction error: {e}")
                response['suggestion']['error'] = str(e)
        
        return JsonResponse(response)
        
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


@require_http_methods(["GET"])
def health_check(request):
    """Health check endpoint"""
    return JsonResponse({
        'status': 'healthy',
        'models_loaded': {
            'outcome_model': outcome_model is not None,
            'lstm_predictor': lstm_predictor is not None,
            'preprocessor': preprocessor is not None,
            'event_log': event_log_df is not None
        }
    })


# URL patterns (to be used in urls.py)
urlpatterns = [
    # path('api/predict/outcome/', predict_outcome, name='predict_outcome'),
    # path('api/predict/next-activity/', predict_next_activity, name='predict_next_activity'),
    # path('api/predict/remaining-time/', predict_remaining_time, name='predict_remaining_time'),
    # path('api/predict/all/', predict_all, name='predict_all'),
    # path('api/health/', health_check, name='health_check'),
]
