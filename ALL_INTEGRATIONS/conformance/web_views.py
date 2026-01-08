"""
Conformance Checking Web Views (Group 6)
"""
from django.shortcuts import render, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.http import HttpResponseForbidden

from uploads.models import EventLog
from discovery.models import DiscoveredProcessModel
from .models import ConformanceResult


@login_required
def conformance_view(request, event_log_id):
    """
    Conformance checking page for an event log (Group 6).
    Shows model selector, run button, and results display.
    """
    # Only analysts and admins can access
    if not (request.user.roles.filter(name='Analyst').exists() or 
            request.user.roles.filter(name='Admin').exists()):
        return HttpResponseForbidden('Access denied.')
    
    # Get the event log
    event_log = get_object_or_404(EventLog, pk=event_log_id)
    
    # Get all discovered models for this log
    discovered_models = DiscoveredProcessModel.objects.filter(
        event_log=event_log
    ).select_related('discovered_by').order_by('-discovered_at')
    
    # Get recent conformance results for this log
    recent_results = ConformanceResult.objects.filter(
        event_log=event_log
    ).select_related('discovered_model', 'discovered_model__event_log').order_by('-created_at')[:5]
    
    context = {
        'event_log': event_log,
        'discovered_models': discovered_models,
        'recent_results': recent_results,
        'has_models': discovered_models.exists(),
    }
    
    return render(request, 'conformance/conformance.html', context)
