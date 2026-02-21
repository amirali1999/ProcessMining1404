"""
Django REST API for Process Mining Predictions
Provides endpoints for outcome, next activity, and remaining time predictions
"""

from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.core.files.storage import FileSystemStorage
from django.utils.timezone import now
import json
import numpy as np
import pandas as pd
import os
import pickle

# Import our models
from .outcome_prediction import OutcomePredictionModel, EnsembleOutcomePredictor
from .torch_models import TorchCombinedPredictor
try:
    from .lstm_models import CombinedLSTMPredictor
except Exception:  # Optional dependency
    CombinedLSTMPredictor = None
from .data_preprocessing import XESDataPreprocessor

# Global variables for loaded models
# Check both project root and local app directory for trained_models
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP_DIR = os.path.dirname(os.path.abspath(__file__))
LOCAL_MODELS_DIR = os.path.join(APP_DIR, 'trained_models')
ROOT_MODELS_DIR = os.path.join(BASE_DIR, 'trained_models')
LOCAL_TORCH_MODELS_DIR = os.path.join(APP_DIR, 'trained_models_torch')
ROOT_TORCH_MODELS_DIR = os.path.join(BASE_DIR, 'trained_models_torch')
UPLOAD_DIR = os.path.join(BASE_DIR, 'uploaded_xes')

outcome_model = None
lstm_predictor = None
preprocessor = None
event_log_df = None
sequence_backend = None
current_xes_path = None


def demo_page(request):
    """Serve the demo page"""
    return render(request, 'demo.html')


def _resolve_default_xes_path():
    env_path = os.environ.get('XES_PATH')
    if env_path and os.path.exists(env_path):
        return env_path

    dataset_dir = os.path.join(BASE_DIR, 'dataset')
    if os.path.isdir(dataset_dir):
        for name in os.listdir(dataset_dir):
            if name.endswith('.xes') or name.endswith('.xes.gz'):
                return os.path.join(dataset_dir, name)

    return None


def _load_event_log(xes_path: str):
    import pm4py
    global event_log_df
    if not xes_path or not os.path.exists(xes_path):
        event_log_df = None
        return

    log = pm4py.read_xes(xes_path)
    event_log_df = pm4py.convert_to_dataframe(log)


def _get_pad_id():
    if preprocessor is None or not hasattr(preprocessor, 'activity_encoder'):
        return 0
    try:
        if 'START' in preprocessor.activity_encoder.classes_:
            return int(np.where(preprocessor.activity_encoder.classes_ == 'START')[0][0])
    except Exception:
        return 0
    return 0

def load_models():
    """Load all trained models"""
    global outcome_model, lstm_predictor, preprocessor, event_log_df, sequence_backend, current_xes_path
    
    print("Loading models...")
    
    # 1. Load preprocessor
    # Check root first (most likely location for shared preprocessor), then local
    preprocessor_path = None
    for base_dir in [LOCAL_TORCH_MODELS_DIR, ROOT_TORCH_MODELS_DIR, ROOT_MODELS_DIR, LOCAL_MODELS_DIR]:
        candidate = os.path.join(base_dir, 'preprocessor.pkl')
        if os.path.exists(candidate):
            preprocessor_path = candidate
            break
        
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
    outcome_model_path = None
    for base_dir in [LOCAL_TORCH_MODELS_DIR, ROOT_TORCH_MODELS_DIR, LOCAL_MODELS_DIR, ROOT_MODELS_DIR]:
        candidate = os.path.join(base_dir, 'ensemble')
        if os.path.exists(candidate):
            outcome_model_path = candidate
            break
        
    if os.path.exists(outcome_model_path):
        outcome_model = EnsembleOutcomePredictor()
        outcome_model.load(outcome_model_path)
        print(f"Outcome model loaded from {outcome_model_path}")
    else:
        print("Warning: Outcome model (ensemble) not found")
    
    # 3. Load LSTM models
    # Check root first (seems to be where they are), then local
    sequence_backend = None
    torch_dir = None
    for base_dir in [LOCAL_TORCH_MODELS_DIR, ROOT_TORCH_MODELS_DIR]:
        candidate = os.path.join(base_dir, 'lstm_torch')
        if os.path.exists(candidate):
            torch_dir = candidate
            break

    if torch_dir and preprocessor is not None:
        try:
            vocab_size = len(preprocessor.activity_encoder.classes_)
            pad_id = _get_pad_id()
            lstm_predictor = TorchCombinedPredictor(vocab_size=vocab_size, pad_id=pad_id)
            lstm_predictor.load(torch_dir)
            sequence_backend = 'torch'
            print(f"Torch LSTM models loaded from {torch_dir}")
        except Exception as e:
            print(f"Warning: Torch models could not be loaded: {e}")
            lstm_predictor = None

    if lstm_predictor is None:
        lstm_model_path = os.path.join(ROOT_MODELS_DIR, 'lstm')
        if not os.path.exists(lstm_model_path) or not os.listdir(lstm_model_path):
            lstm_model_path = os.path.join(LOCAL_MODELS_DIR, 'lstm')

        if os.path.exists(lstm_model_path) and CombinedLSTMPredictor is not None:
            metadata_path = os.path.join(lstm_model_path, 'best_next_activity_model.keras_metadata.pkl')
            if not os.path.exists(metadata_path):
                metadata_path = os.path.join(lstm_model_path, 'next_activity_lstm_metadata.pkl')

            if os.path.exists(metadata_path):
                with open(metadata_path, 'rb') as f:
                    metadata = pickle.load(f)

                lstm_predictor = CombinedLSTMPredictor(
                    vocab_size=metadata['vocab_size'],
                    max_length=metadata['max_length']
                )
                lstm_predictor.load(lstm_model_path)
                sequence_backend = 'tf'
                print(f"LSTM models loaded from {lstm_model_path}")
            elif preprocessor is not None:
                print("Metadata not found, inferring from preprocessor...")
                vocab_size = len(preprocessor.activity_encoder.classes_)
                lstm_predictor = CombinedLSTMPredictor(
                    vocab_size=vocab_size,
                    max_length=50
                )
                lstm_predictor.load(lstm_model_path)
                sequence_backend = 'tf'
                print(f"LSTM models loaded (inferred metadata) from {lstm_model_path}")
        else:
            print("Warning: LSTM models not found")
    
    # 4. Load event log for case lookup
    # Use BASE_DIR to find the event log folder
    if current_xes_path and os.path.exists(current_xes_path):
        _load_event_log(current_xes_path)
        if event_log_df is not None:
            print(f"Event log loaded: {len(event_log_df)} events")
    else:
        xes_file = _resolve_default_xes_path()
        if xes_file:
            _load_event_log(xes_file)
            if event_log_df is not None:
                print(f"Event log loaded: {len(event_log_df)} events")
        else:
            print("Warning: Event log not found")
    
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
    
    pad_id = _get_pad_id()

    # Encode activities
    encoded_activities = []
    for act in activities:
        try:
            encoded = preprocessor.activity_encoder.transform([act])[0]
        except ValueError:
            # Unknown activity - use pad_id
            encoded = pad_id
        encoded_activities.append(encoded)
    
    # Pad or truncate
    if len(encoded_activities) < max_length:
        # Pad with pad_id
        encoded_activities = [pad_id] * (max_length - len(encoded_activities)) + encoded_activities
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
        
        if preprocessor is None:
            return JsonResponse({'error': 'Preprocessor not loaded'}, status=500)

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
        
        if preprocessor is None:
            return JsonResponse({'error': 'Preprocessor not loaded'}, status=500)

        # Get case data
        case_data = get_case_data(case_id)
        if case_data is None:
            return JsonResponse({'error': f'Case {case_id} not found'}, status=404)
        
        # Prepare sequence
        sequence = prepare_sequence(case_data['activities'])
        
        # Make prediction
        if lstm_predictor is None:
            return JsonResponse({'error': 'LSTM model not loaded'}, status=500)
        
        if sequence_backend == 'torch':
            next_activity_encoded = lstm_predictor.predict_next(sequence)[0]
            next_activity = preprocessor.activity_encoder.inverse_transform([next_activity_encoded])[0]

            proba = lstm_predictor.predict_next(sequence, return_proba=True)[0]
            top_k_indices = np.argsort(proba)[-5:][::-1]
            top_k_activities = preprocessor.activity_encoder.inverse_transform(top_k_indices)
            top_k_proba = proba[top_k_indices]
        else:
            next_activity_encoded = lstm_predictor.next_activity_model.predict(sequence)[0]
            next_activity = preprocessor.activity_encoder.inverse_transform([next_activity_encoded])[0]

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
        
        if preprocessor is None:
            return JsonResponse({'error': 'Preprocessor not loaded'}, status=500)

        # Get case data
        case_data = get_case_data(case_id)
        if case_data is None:
            return JsonResponse({'error': f'Case {case_id} not found'}, status=404)
        
        # Prepare sequence
        sequence = prepare_sequence(case_data['activities'])
        
        # Make prediction
        if lstm_predictor is None:
            return JsonResponse({'error': 'LSTM model not loaded'}, status=500)
        
        if sequence_backend == 'torch':
            remaining_time_norm = lstm_predictor.predict_remaining_time(sequence)[0]
        else:
            remaining_time_norm = lstm_predictor.remaining_time_model.predict(sequence)[0]
        
        # Denormalize time
        if hasattr(preprocessor, 'time_scaler'):
            if isinstance(remaining_time_norm, (list, np.ndarray)):
                val = remaining_time_norm[0] if len(remaining_time_norm) > 0 else 0
            else:
                val = remaining_time_norm
            log_time = preprocessor.time_scaler.inverse_transform([[val]])[0][0]
            remaining_time_seconds = np.expm1(log_time)
        else:
            remaining_time_seconds = remaining_time_norm
        
        # Calculate elapsed time
        timestamps = case_data.get('timestamps', [])
        elapsed_time = (timestamps[-1] - timestamps[0]).total_seconds() if timestamps else 0
        
        response = {
            'case_id': case_id,
            'predicted_remaining_time_seconds': float(remaining_time_seconds),
            'predicted_remaining_time_minutes': float(remaining_time_seconds / 60),
            'predicted_remaining_time_hours': float(remaining_time_seconds / 3600),
            'predicted_remaining_time_days': float(remaining_time_seconds / 86400),
            'elapsed_time_seconds': elapsed_time,
            'current_activities': case_data['activities']
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
        
        if preprocessor is None:
            return JsonResponse({'error': 'Preprocessor not loaded'}, status=500)

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
                if sequence_backend == 'torch':
                    next_activity_encoded = lstm_predictor.predict_next(sequence)[0]
                    next_activity = preprocessor.activity_encoder.inverse_transform([next_activity_encoded])[0]
                    response['suggestion']['next_activity'] = next_activity

                    remaining_time_norm = lstm_predictor.predict_remaining_time(sequence)[0]
                else:
                    next_activity_encoded = lstm_predictor.next_activity_model.predict(sequence)[0]
                    next_activity = preprocessor.activity_encoder.inverse_transform([next_activity_encoded])[0]
                    response['suggestion']['next_activity'] = next_activity

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
        'sequence_backend': sequence_backend,
        'models_loaded': {
            'outcome_model': outcome_model is not None,
            'lstm_predictor': lstm_predictor is not None,
            'preprocessor': preprocessor is not None,
            'event_log': event_log_df is not None
        }
    })


@csrf_exempt
@require_http_methods(["POST"])
def upload_and_train(request):
    """
    Upload XES and train models.

    POST /api/train/
    Form-Data:
      - xes_file: file
      - train_outcome: true/false
      - train_lstm: true/false
      - max_case_length: int (optional)
      - batch_size: int (optional)
      - epochs_next: int (optional)
      - epochs_time: int (optional)
    """
    global outcome_model, lstm_predictor, preprocessor, current_xes_path, sequence_backend

    if 'xes_file' not in request.FILES:
        return JsonResponse({'error': 'xes_file is required'}, status=400)

    train_outcome = str(request.POST.get('train_outcome', 'true')).lower() in ['true', '1', 'yes', 'on']
    train_lstm = str(request.POST.get('train_lstm', 'true')).lower() in ['true', '1', 'yes', 'on']
    if not train_outcome and not train_lstm:
        return JsonResponse({'error': 'At least one of train_outcome or train_lstm must be true'}, status=400)

    max_case_length = int(request.POST.get('max_case_length', 50))
    batch_size = int(request.POST.get('batch_size', 512))
    epochs_next = int(request.POST.get('epochs_next', 5))
    epochs_time = int(request.POST.get('epochs_time', 5))

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    storage = FileSystemStorage(location=UPLOAD_DIR)
    xes_file = request.FILES['xes_file']
    timestamp = now().strftime('%Y%m%d_%H%M%S')
    filename = storage.save(f"{timestamp}_{xes_file.name}", xes_file)
    file_path = storage.path(filename)

    current_xes_path = file_path

    output_dir = LOCAL_TORCH_MODELS_DIR
    os.makedirs(output_dir, exist_ok=True)

    try:
        preprocessor = XESDataPreprocessor(file_path)
        preprocessor.load_xes()
        preprocessor.convert_to_dataframe()
        preprocessor.clean_data()

        if train_outcome:
            X_train, X_test, y_train, y_test, feature_columns = preprocessor.prepare_outcome_prediction_data()
            ensemble = EnsembleOutcomePredictor()
            ensemble.train(X_train, y_train, feature_columns)
            ensemble.evaluate(X_test, y_test, preprocessor.outcome_encoder)
            ensemble_dir = os.path.join(output_dir, 'ensemble')
            ensemble.save(ensemble_dir)
            outcome_model = ensemble

        if train_lstm:
            lstm_data = preprocessor.build_torch_dataloader_for_lstm(
                max_case_length=max_case_length,
                batch_size=batch_size,
                time_mode="log1p",
                num_workers=0
            )
            predictor = TorchCombinedPredictor(
                vocab_size=lstm_data['vocab_size'],
                pad_id=lstm_data['pad_id']
            )
            predictor.train_next(lstm_data['loader'], epochs=epochs_next, lr=1e-3)
            predictor.train_time(lstm_data['loader'], epochs=epochs_time, lr=1e-3)
            lstm_dir = os.path.join(output_dir, 'lstm_torch')
            predictor.save(lstm_dir)
            lstm_predictor = predictor
            sequence_backend = 'torch'

        preprocessor_path = os.path.join(output_dir, 'preprocessor.pkl')
        preprocessor.save_preprocessor(preprocessor_path)

        info_path = os.path.join(output_dir, 'training_info.txt')
        with open(info_path, 'w') as f:
            f.write(f"Training completed at: {now()}\n")
            f.write(f"XES file: {file_path}\n")
            f.write(f"Number of cases: {preprocessor.df['case:concept:name'].nunique()}\n")
            f.write(f"Number of events: {len(preprocessor.df)}\n")
            f.write(f"Unique activities: {preprocessor.df['concept:name'].nunique()}\n")
            if train_outcome:
                f.write(f"Outcome classes: {len(preprocessor.outcome_encoder.classes_)}\n")
            if train_lstm:
                f.write(f"Activity vocabulary size: {len(preprocessor.activity_encoder.classes_)}\n")

        _load_event_log(file_path)

        return JsonResponse({
            'status': 'ok',
            'xes_path': file_path,
            'models_dir': output_dir,
            'trained': {
                'outcome': train_outcome,
                'lstm_torch': train_lstm
            }
        })
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


# URL patterns (to be used in urls.py)
urlpatterns = [
    # path('api/predict/outcome/', predict_outcome, name='predict_outcome'),
    # path('api/predict/next-activity/', predict_next_activity, name='predict_next_activity'),
    # path('api/predict/remaining-time/', predict_remaining_time, name='predict_remaining_time'),
    # path('api/predict/all/', predict_all, name='predict_all'),
    # path('api/health/', health_check, name='health_check'),
]
