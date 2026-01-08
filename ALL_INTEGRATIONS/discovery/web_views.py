from django.shortcuts import render, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.http import HttpResponseForbidden

from uploads.models import EventLog
from .models import DiscoveredProcessModel
from . import services
from preprocessing.services import _compute_log_stats, get_event_log_dataframe


@login_required
def discovery_dashboard(request):
    """
    Dashboard showing all event logs and their discovered models.
    """
    event_logs = EventLog.objects.all().select_related('uploaded_file').prefetch_related('discovered_models')
    
    context = {
        'event_logs': event_logs,
    }
    
    return render(request, 'discovery/dashboard.html', context)


@login_required
def discover_view(request, event_log_id):
    """
    Discovery page for a specific event log.
    Shows algorithm buttons and discovered models.
    """
    event_log = get_object_or_404(EventLog.objects.select_related('uploaded_file'), pk=event_log_id)
    
    # Get all discovered models for this log
    discovered_models = services.get_discovered_models(event_log_id)
    
    # Compute statistics for RAW and CLEANED versions
    raw_stats = None
    cleaned_stats = None
    
    try:
        df_raw = get_event_log_dataframe(event_log_id, version="raw")
        raw_stats = _compute_log_stats(df_raw)
    except Exception:
        pass
    
    if event_log.has_cleaned_version:
        try:
            df_cleaned = get_event_log_dataframe(event_log_id, version="cleaned")
            cleaned_stats = _compute_log_stats(df_cleaned)
        except Exception:
            pass
    
    context = {
        'event_log': event_log,
        'discovered_models': discovered_models,
        'has_cleaned': event_log.has_cleaned_version,
        'raw_stats': raw_stats,
        'cleaned_stats': cleaned_stats,
    }
    
    return render(request, 'discovery/discover.html', context)


@login_required
def visualize_view(request, model_id):
    """
    Visualization page for a discovered process model (Group 5).
    Shows the Petri net diagram with download options.
    """
    # Only analysts and admins can access
    if not (request.user.roles.filter(name='Analyst').exists() or 
            request.user.roles.filter(name='Admin').exists()):
        return HttpResponseForbidden('Access denied.')
    
    # Get the discovered model
    model = get_object_or_404(
        DiscoveredProcessModel.objects.select_related('event_log', 'discovered_by'),
        pk=model_id
    )
    
    context = {
        'model': model,
        'event_log': model.event_log,
    }
    
    return render(request, 'discovery/visualize.html', context)
