"""
Conformance Checking REST API (Group 6)

Provides endpoints for token replay conformance checking.
"""
from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from django.shortcuts import get_object_or_404

from .models import ConformanceResult
from . import services


class ConformanceViewSet(viewsets.ViewSet):
    """
    API endpoints for conformance checking (Group 6).
    
    Endpoints:
    - POST /api/conformance/run/ - Run token replay
    - GET /api/conformance/results/{id}/ - Get result summary
    - GET /api/conformance/results/{id}/cases/ - Get case details
    """
    permission_classes = [IsAuthenticated]
    
    @action(detail=False, methods=['post'], url_path='run')
    def run_conformance(self, request):
        """
        Run token replay conformance checking.
        
        POST /api/conformance/run/
        
        Body:
        {
            "event_log_id": int,
            "discovered_model_id": int,
            "source": "default"|"raw"|"cleaned"  (optional, default="default")
        }
        
        Returns conformance result with statistics.
        """
        event_log_id = request.data.get('event_log_id')
        discovered_model_id = request.data.get('discovered_model_id')
        source = request.data.get('source', 'default')
        
        if not event_log_id:
            return Response(
                {'error': 'event_log_id is required'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        if not discovered_model_id:
            return Response(
                {'error': 'discovered_model_id is required'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        if source not in ['default', 'raw', 'cleaned']:
            return Response(
                {'error': 'source must be "default", "raw", or "cleaned"'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        try:
            # Run conformance checking
            result = services.run_token_replay_conformance(
                event_log_id=event_log_id,
                discovered_model_id=discovered_model_id,
                source=source
            )
            
            return Response(result, status=status.HTTP_201_CREATED)
            
        except Exception as e:
            import traceback
            print(f"Error running conformance: {str(e)}")
            print(traceback.format_exc())
            return Response(
                {'error': f'Failed to run conformance check: {str(e)}'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
    
    @action(detail=True, methods=['get'], url_path='summary')
    def get_result(self, request, pk=None):
        """
        Get conformance result summary.
        
        GET /api/conformance/results/{id}/summary/
        
        Returns the same structure as run_conformance.
        """
        try:
            result = get_object_or_404(ConformanceResult, pk=pk)
            
            return Response({
                "conformance_result_id": result.id,
                "event_log_id": result.event_log.id,
                "event_log_name": result.event_log.name,
                "discovered_model_id": result.discovered_model.id,
                "discovered_model_name": str(result.discovered_model),
                "source": result.source_version,
                "created_at": result.created_at,
                "stats": {
                    "total_cases": result.total_cases,
                    "compliant_cases": result.compliant_cases,
                    "non_compliant_cases": result.non_compliant_cases,
                    "compliant_percentage": round(result.compliant_percentage, 2),
                    "non_compliant_percentage": round(result.non_compliant_percentage, 2)
                }
            })
            
        except Exception as e:
            return Response(
                {'error': f'Failed to retrieve result: {str(e)}'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
    
    @action(detail=True, methods=['get'], url_path='cases')
    def get_cases(self, request, pk=None):
        """
        Get paginated list of compliant or non-compliant cases.
        
        GET /api/conformance/results/{id}/cases/?status=compliant&page=1&page_size=50
        
        Query Parameters:
        - status: "compliant" or "non_compliant" (required)
        - page: page number (default=1)
        - page_size: items per page (default=50, max=200)
        
        Returns paginated case data (log rows).
        """
        case_status = request.query_params.get('status')
        if not case_status:
            return Response(
                {'error': 'status query parameter is required ("compliant" or "non_compliant")'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        if case_status not in ['compliant', 'non_compliant']:
            return Response(
                {'error': 'status must be "compliant" or "non_compliant"'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        try:
            page = int(request.query_params.get('page', 1))
            page_size = min(int(request.query_params.get('page_size', 50)), 200)
        except ValueError:
            return Response(
                {'error': 'page and page_size must be integers'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        try:
            result = services.get_conformance_cases(
                conformance_result_id=pk,
                status=case_status,
                page=page,
                page_size=page_size
            )
            
            return Response(result)
            
        except ConformanceResult.DoesNotExist:
            return Response(
                {'error': f'Conformance result {pk} not found'},
                status=status.HTTP_404_NOT_FOUND
            )
        except Exception as e:
            import traceback
            print(f"Error retrieving cases: {str(e)}")
            print(traceback.format_exc())
            return Response(
                {'error': f'Failed to retrieve cases: {str(e)}'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
