"""
DRF ViewSet for Prediction API
===============================
REST API endpoints for process predictions.

Endpoints:
- POST /api/event-logs/{id}/prediction/all/ - Run all predictions
- POST /api/event-logs/{id}/prediction/outcome/ - Predict outcome only
- POST /api/event-logs/{id}/prediction/next-activity/ - Predict next activity
- POST /api/event-logs/{id}/prediction/remaining-time/ - Predict remaining time
- GET /api/prediction/health/ - Health check for model loading
"""

from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from django.shortcuts import get_object_or_404, render
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse

from uploads.models import EventLog
from .serializers import PredictionInputSerializer
from . import services


class PredictionViewSet(viewsets.ViewSet):
    """
    ViewSet for process prediction endpoints.
    
    Uses pre-trained Group 7 models for:
    - Outcome prediction (ensemble classifier)
    - Next activity prediction (LSTM)
    - Remaining time prediction (LSTM)
    """
    permission_classes = [IsAuthenticated]
    
    @action(detail=True, methods=['post'], url_path='prediction/all')
    def predict_all(self, request, pk=None):
        """
        Run all predictions for a case.
        
        POST /api/event-logs/{id}/prediction/all/
        Body: {"source": "default", "case_id": "case_123"}
        or: {"source": "default", "activities": ["A", "B", "C"]}
        """
        event_log = get_object_or_404(EventLog, pk=pk)
        
        serializer = PredictionInputSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        
        try:
            result = services.predict_all(
                event_log_id=event_log.id,
                source=serializer.validated_data.get('source', 'default'),
                case_id=serializer.validated_data.get('case_id'),
                activities=serializer.validated_data.get('activities')
            )
            return Response(result, status=status.HTTP_200_OK)
        except Exception as e:
            return Response(
                {'error': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
    
    @action(detail=True, methods=['post'], url_path='prediction/outcome')
    def predict_outcome(self, request, pk=None):
        """
        Predict outcome for a case.
        
        POST /api/event-logs/{id}/prediction/outcome/
        Body: {"source": "default", "case_id": "case_123"}
        """
        event_log = get_object_or_404(EventLog, pk=pk)
        
        serializer = PredictionInputSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        
        try:
            result = services.predict_outcome(
                event_log_id=event_log.id,
                source=serializer.validated_data.get('source', 'default'),
                case_id=serializer.validated_data.get('case_id')
            )
            return Response(result, status=status.HTTP_200_OK)
        except Exception as e:
            return Response(
                {'error': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
    
    @action(detail=True, methods=['post'], url_path='prediction/next-activity')
    def predict_next_activity(self, request, pk=None):
        """
        Predict next activity.
        
        POST /api/event-logs/{id}/prediction/next-activity/
        Body: {"source": "default", "case_id": "case_123"}
        or: {"source": "default", "activities": ["A", "B", "C"]}
        """
        event_log = get_object_or_404(EventLog, pk=pk)
        
        serializer = PredictionInputSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        
        try:
            result = services.predict_next_activity(
                event_log_id=event_log.id,
                source=serializer.validated_data.get('source', 'default'),
                case_id=serializer.validated_data.get('case_id'),
                activities=serializer.validated_data.get('activities')
            )
            return Response(result, status=status.HTTP_200_OK)
        except Exception as e:
            return Response(
                {'error': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
    
    @action(detail=True, methods=['post'], url_path='prediction/remaining-time')
    def predict_remaining_time(self, request, pk=None):
        """
        Predict remaining time.
        
        POST /api/event-logs/{id}/prediction/remaining-time/
        Body: {"source": "default", "case_id": "case_123"}
        or: {"source": "default", "activities": ["A", "B", "C"]}
        """
        event_log = get_object_or_404(EventLog, pk=pk)
        
        serializer = PredictionInputSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        
        try:
            result = services.predict_remaining_time(
                event_log_id=event_log.id,
                source=serializer.validated_data.get('source', 'default'),
                case_id=serializer.validated_data.get('case_id'),
                activities=serializer.validated_data.get('activities')
            )
            return Response(result, status=status.HTTP_200_OK)
        except Exception as e:
            return Response(
                {'error': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
    
    @action(detail=True, methods=['get'], url_path='prediction/cases')
    def get_cases(self, request, pk=None):
        """
        Get list of case IDs from an event log.
        
        GET /api/event-logs/{id}/prediction/cases/?source=default
        """
        event_log = get_object_or_404(EventLog, pk=pk)
        source = request.query_params.get('source', 'default')
        
        try:
            df = services.get_log_for_prediction(event_log.id, source)
            case_ids = df['case:concept:name'].unique().tolist()
            
            return Response({
                'case_ids': case_ids,
                'count': len(case_ids)
            }, status=status.HTTP_200_OK)
        except Exception as e:
            return Response(
                {'error': f'Failed to load cases: {str(e)}'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
    
    @action(detail=False, methods=['get'], url_path='health')
    def health_check(self, request):
        """
        Check if prediction models are loaded.
        
        GET /api/prediction/health/
        """
        try:
            models = services.load_prediction_models()
            return Response({
                'status': 'healthy',
                'models_loaded': True,
                'vocab_size': models['vocab_size'],
                'max_length': models['max_length']
            }, status=status.HTTP_200_OK)
        except Exception as e:
            return Response({
                'status': 'unhealthy',
                'models_loaded': False,
                'error': str(e)
            }, status=status.HTTP_503_SERVICE_UNAVAILABLE)


# Standalone API view for getting cases
def get_cases_view(request, pk):
    """
    Get list of case IDs from an event log.
    
    GET /api/event-logs/{id}/prediction/cases/?source=default
    """
    if not request.user.is_authenticated:
        return JsonResponse({'error': 'Authentication required'}, status=401)
    
    event_log = get_object_or_404(EventLog, pk=pk)
    source = request.GET.get('source', 'default')
    
    try:
        df = services.get_log_for_prediction(event_log.id, source)
        case_ids = df['case:concept:name'].unique().tolist()
        
        return JsonResponse({
            'case_ids': case_ids,
            'count': len(case_ids)
        })
    except Exception as e:
        return JsonResponse(
            {'error': f'Failed to load cases: {str(e)}'},
            status=500
        )


# Web View for Prediction Page
@login_required
def prediction_page(request):
    """Render the prediction UI page"""
    return render(request, 'prediction/prediction.html')
