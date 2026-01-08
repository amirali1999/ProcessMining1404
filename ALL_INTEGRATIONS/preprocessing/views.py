from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from rest_framework.authentication import SessionAuthentication
from django.shortcuts import get_object_or_404

from uploads.models import EventLog
from .serializers import (
    EventLogListSerializer,
    EventLogDetailSerializer,
    DefaultSourceSerializer,
)
from .services import (
    smart_clean_event_log,
    get_event_log_table_data,
)


class EventLogViewSet(viewsets.ReadOnlyModelViewSet):
    """
    API ViewSet for Event Logs.
    
    Provides:
    - GET /api/event-logs/ - List all event logs
    - GET /api/event-logs/{id}/ - Retrieve single event log
    - POST /api/event-logs/{id}/smart-clean/ - Trigger Smart Clean
    - GET /api/event-logs/{id}/table/ - Get paginated table data
    - PATCH /api/event-logs/{id}/default-source/ - Update default source
    """
    queryset = EventLog.objects.select_related('uploaded_file').all()
    authentication_classes = [SessionAuthentication]
    permission_classes = [IsAuthenticated]
    
    def get_serializer_class(self):
        if self.action == 'list':
            return EventLogListSerializer
        return EventLogDetailSerializer
    
    @action(detail=True, methods=['post'])
    def smart_clean(self, request, pk=None):
        """
        Trigger Smart Clean on an event log.
        
        POST /api/event-logs/{id}/smart-clean/
        
        Optional body parameters:
        - aggressive: bool (default: false)
        - normalize_names: bool (default: true)
        """
        event_log = self.get_object()
        
        # Get parameters
        aggressive = request.data.get('aggressive', False)
        normalize_names = request.data.get('normalize_names', True)
        
        try:
            result = smart_clean_event_log(
                event_log_id=event_log.id,
                aggressive=aggressive,
                normalize_names=normalize_names
            )
            
            return Response({
                'status': 'success',
                'message': 'Smart Clean completed successfully',
                'data': result,
            }, status=status.HTTP_200_OK)
            
        except Exception as e:
            return Response({
                'status': 'error',
                'message': str(e),
            }, status=status.HTTP_400_BAD_REQUEST)
    
    @action(detail=True, methods=['get'])
    def table(self, request, pk=None):
        """
        Get paginated table data for display.
        
        GET /api/event-logs/{id}/table/?version=raw&page=1&page_size=50
        
        Query parameters:
        - version: "raw" or "cleaned" (default: raw)
        - page: page number (default: 1)
        - page_size: rows per page (default: 50)
        """
        event_log = self.get_object()
        
        # Get parameters
        version = request.query_params.get('version', 'raw')
        page = int(request.query_params.get('page', 1))
        page_size = int(request.query_params.get('page_size', 50))
        
        # Validate version
        if version not in ['raw', 'cleaned']:
            return Response({
                'status': 'error',
                'message': 'Invalid version. Must be "raw" or "cleaned".',
            }, status=status.HTTP_400_BAD_REQUEST)
        
        try:
            table_data = get_event_log_table_data(
                event_log_id=event_log.id,
                version=version,
                page=page,
                page_size=page_size
            )
            
            return Response({
                'status': 'success',
                'data': table_data,
            }, status=status.HTTP_200_OK)
            
        except ValueError as e:
            return Response({
                'status': 'error',
                'message': str(e),
            }, status=status.HTTP_422_UNPROCESSABLE_ENTITY)
        except Exception as e:
            return Response({
                'status': 'error',
                'message': f'Failed to load table data: {str(e)}',
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
    @action(detail=True, methods=['patch'])
    def default_source(self, request, pk=None):
        """
        Update the default source for downstream modules.
        
        PATCH /api/event-logs/{id}/default-source/
        
        Body:
        {
            "default_source_for_downstream": "raw" | "cleaned"
        }
        """
        event_log = self.get_object()
        
        serializer = DefaultSourceSerializer(data=request.data)
        if serializer.is_valid():
            source = serializer.validated_data['default_source_for_downstream']
            
            # Check if cleaned version exists when setting to cleaned
            if source == 'cleaned' and not event_log.has_cleaned_version:
                return Response({
                    'status': 'error',
                    'message': 'Cannot set default to "cleaned" - no cleaned version exists. Run Smart Clean first.',
                }, status=status.HTTP_400_BAD_REQUEST)
            
            event_log.default_source_for_downstream = source
            event_log.save()
            
            return Response({
                'status': 'success',
                'message': f'Default source updated to {source}',
                'data': EventLogDetailSerializer(event_log).data,
            }, status=status.HTTP_200_OK)
        
        return Response({
            'status': 'error',
            'errors': serializer.errors,
        }, status=status.HTTP_400_BAD_REQUEST)
    
    @action(detail=True, methods=['get'])
    def stats(self, request, pk=None):
        """
        Get statistics for both RAW and CLEANED versions of the log.
        
        GET /api/event-logs/{id}/stats/
        
        Returns:
        {
            "status": "success",
            "data": {
                "raw": {"num_cases": 1050, "num_events": 15214, ...},
                "cleaned": {"num_cases": 1000, "num_events": 14500, ...}
            }
        }
        """
        from .services import get_event_log_dataframe, _compute_log_stats
        
        event_log = self.get_object()
        
        try:
            # Get RAW statistics
            raw_df = get_event_log_dataframe(event_log, version='raw')
            raw_stats = _compute_log_stats(raw_df)
            
            # Get CLEANED statistics (if available)
            cleaned_stats = None
            if event_log.has_cleaned_version:
                try:
                    cleaned_df = get_event_log_dataframe(event_log, version='cleaned')
                    cleaned_stats = _compute_log_stats(cleaned_df)
                except Exception:
                    cleaned_stats = None
            
            return Response({
                'status': 'success',
                'data': {
                    'raw': raw_stats,
                    'cleaned': cleaned_stats,
                },
            }, status=status.HTTP_200_OK)
            
        except Exception as e:
            return Response({
                'status': 'error',
                'message': str(e),
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
