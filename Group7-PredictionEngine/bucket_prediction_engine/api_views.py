"""
گام ۶: Django REST API for Predictive Process Monitoring
Group 7 - Prediction Engine API

Endpoints:
- POST /api/predict/next-activity/
- POST /api/predict/remaining-time/
- POST /api/predict/outcome/
- POST /api/predict/all/
"""

from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
import joblib
import numpy as np
import os
import pandas as pd


# Global variables for loaded models
MODELS_LOADED = False
activity_encoder = None
outcome_predictor = None
bucket_ensemble = None
event_log_df = None


def load_models(models_dir='trained_models'):
    """
    Load all trained models
    
    Args:
        models_dir: Directory containing trained models
    """
    global MODELS_LOADED, activity_encoder, outcome_predictor, bucket_ensemble
    
    if MODELS_LOADED:
        return
    
    print("Loading models...")
    
    # Load activity encoder
    activity_encoder = joblib.load(os.path.join(models_dir, 'activity_encoder.pkl'))
    
    # Load outcome predictor
    from outcome_models import OutcomePredictor
    outcome_predictor = OutcomePredictor(activity_encoder)
    outcome_predictor.load(os.path.join(models_dir, 'outcome'))
    
    # Load bucket ensemble
    from bucket_models import BucketEnsemble
    bucket_ensemble = BucketEnsemble(vocab_size=activity_encoder.vocab_size)
    bucket_ensemble.load(os.path.join(models_dir, 'buckets'))
    
    MODELS_LOADED = True
    print("✅ Models loaded successfully")


def load_event_log(xes_path):
    """
    Load event log for case lookup
    
    Args:
        xes_path: Path to XES file
    """
    global event_log_df
    
    if event_log_df is not None:
        return
    
    import pm4py
    
    print(f"Loading event log from {xes_path}...")
    log = pm4py.read_xes(xes_path)
    event_log_df = pm4py.convert_to_dataframe(log)
    
    # Standardize column names
    event_log_df.rename(columns={
        'case:concept:name': 'case_id',
        'concept:name': 'activity',
        'time:timestamp': 'timestamp'
    }, inplace=True)
    
    event_log_df = event_log_df.sort_values(['case_id', 'timestamp'])
    
    print(f"✅ Loaded {len(event_log_df)} events")


def get_case_prefix(case_id):
    """
    گام ۹.۲: Get current prefix for a case
    
    Args:
        case_id: Case identifier
        
    Returns:
        List of activities in order
    """
    global event_log_df
    
    if event_log_df is None:
        raise ValueError("Event log not loaded")
    
    # Get events for this case
    case_events = event_log_df[event_log_df['case_id'] == case_id]
    
    if len(case_events) == 0:
        raise ValueError(f"Case {case_id} not found")
    
    # Sort by timestamp and extract activities
    case_events = case_events.sort_values('timestamp')
    activities = case_events['activity'].tolist()
    
    return activities


@api_view(['POST'])
def predict_next_activity(request):
    """
    گام ۹.۱: Predict next activity for a case
    
    POST /api/predict/next-activity/
    {
        "case_id": "H12345"
    }
    
    or
    
    {
        "prefix": ["Registration", "Triage", "XRay"]
    }
    """
    load_models()
    
    # Get input
    case_id = request.data.get('case_id')
    prefix = request.data.get('prefix')
    
    # گام ۹.۲ steps 1-3: Get prefix from case_id or use provided prefix
    if case_id:
        try:
            prefix = get_case_prefix(case_id)
        except Exception as e:
            return Response(
                {'error': str(e)},
                status=status.HTTP_404_NOT_FOUND
            )
    elif not prefix:
        return Response(
            {'error': 'Either case_id or prefix must be provided'},
            status=status.HTTP_400_BAD_REQUEST
        )
    
    # گام ۹.۲ step 4-5: Encode prefix
    encoded_prefix = activity_encoder.encode_prefix(prefix)
    
    # گام ۹.۲ step 6-7: Predict using bucket ensemble
    try:
        prediction = bucket_ensemble.predict(encoded_prefix, activity_encoder)
    except Exception as e:
        return Response(
            {'error': f'Prediction failed: {str(e)}'},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )
    
    # گام ۹.۲ step 8: Return result
    response = {
        'case_id': case_id,
        'current_prefix': prefix,
        'predicted_next_activity': prediction['predicted_next_activity'],
        'top_k_candidates': prediction['top_k_candidates'],
        'bucket_id': prediction['bucket_id']
    }
    
    return Response(response, status=status.HTTP_200_OK)


@api_view(['POST'])
def predict_remaining_time(request):
    """
    Predict remaining time for a case
    
    POST /api/predict/remaining-time/
    {
        "case_id": "H12345"
    }
    """
    load_models()
    
    # Get input
    case_id = request.data.get('case_id')
    prefix = request.data.get('prefix')
    
    if case_id:
        try:
            prefix = get_case_prefix(case_id)
        except Exception as e:
            return Response(
                {'error': str(e)},
                status=status.HTTP_404_NOT_FOUND
            )
    elif not prefix:
        return Response(
            {'error': 'Either case_id or prefix must be provided'},
            status=status.HTTP_400_BAD_REQUEST
        )
    
    # Encode and predict
    encoded_prefix = activity_encoder.encode_prefix(prefix)
    
    try:
        prediction = bucket_ensemble.predict(encoded_prefix, activity_encoder)
    except Exception as e:
        return Response(
            {'error': f'Prediction failed: {str(e)}'},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )
    
    response = {
        'case_id': case_id,
        'current_prefix': prefix,
        'predicted_remaining_time_hours': prediction['predicted_remaining_time'],
        'bucket_id': prediction['bucket_id']
    }
    
    return Response(response, status=status.HTTP_200_OK)


@api_view(['POST'])
def predict_outcome(request):
    """
    Predict final outcome for a case
    
    POST /api/predict/outcome/
    {
        "case_id": "H12345"
    }
    """
    load_models()
    
    # Get input
    case_id = request.data.get('case_id')
    prefix = request.data.get('prefix')
    
    if case_id:
        try:
            prefix = get_case_prefix(case_id)
        except Exception as e:
            return Response(
                {'error': str(e)},
                status=status.HTTP_404_NOT_FOUND
            )
    elif not prefix:
        return Response(
            {'error': 'Either case_id or prefix must be provided'},
            status=status.HTTP_400_BAD_REQUEST
        )
    
    # Create prefix sample
    prefix_sample = {
        'prefix_activities': prefix,
        'prefix_length': len(prefix)
    }
    
    # Predict
    try:
        outcome, confidence = outcome_predictor.predict(prefix_sample)
    except Exception as e:
        return Response(
            {'error': f'Prediction failed: {str(e)}'},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )
    
    response = {
        'case_id': case_id,
        'current_prefix': prefix,
        'predicted_outcome': outcome,
        'confidence': float(confidence)
    }
    
    return Response(response, status=status.HTTP_200_OK)


@api_view(['POST'])
def predict_all(request):
    """
    Make all predictions at once
    
    POST /api/predict/all/
    {
        "case_id": "H12345"
    }
    """
    load_models()
    
    # Get input
    case_id = request.data.get('case_id')
    prefix = request.data.get('prefix')
    
    if case_id:
        try:
            prefix = get_case_prefix(case_id)
        except Exception as e:
            return Response(
                {'error': str(e)},
                status=status.HTTP_404_NOT_FOUND
            )
    elif not prefix:
        return Response(
            {'error': 'Either case_id or prefix must be provided'},
            status=status.HTTP_400_BAD_REQUEST
        )
    
    # Encode prefix
    encoded_prefix = activity_encoder.encode_prefix(prefix)
    
    # Predict next activity and remaining time
    try:
        seq_prediction = bucket_ensemble.predict(encoded_prefix, activity_encoder)
    except Exception as e:
        seq_prediction = {'error': str(e)}
    
    # Predict outcome
    try:
        prefix_sample = {'prefix_activities': prefix, 'prefix_length': len(prefix)}
        outcome, confidence = outcome_predictor.predict(prefix_sample)
    except Exception as e:
        outcome = None
        confidence = 0.0
    
    response = {
        'case_id': case_id,
        'current_prefix': prefix,
        'predictions': {
            'next_activity': {
                'activity': seq_prediction.get('predicted_next_activity'),
                'top_k_candidates': seq_prediction.get('top_k_candidates', [])
            },
            'remaining_time': {
                'hours': seq_prediction.get('predicted_remaining_time'),
                'bucket_id': seq_prediction.get('bucket_id')
            },
            'outcome': {
                'prediction': outcome,
                'confidence': float(confidence)
            }
        }
    }
    
    return Response(response, status=status.HTTP_200_OK)


@api_view(['GET'])
def health_check(request):
    """
    Health check endpoint
    
    GET /api/health/
    """
    return Response({
        'status': 'ok',
        'models_loaded': MODELS_LOADED,
        'event_log_loaded': event_log_df is not None
    }, status=status.HTTP_200_OK)
