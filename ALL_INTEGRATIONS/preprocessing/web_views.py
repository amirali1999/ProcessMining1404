from django.shortcuts import render, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.http import HttpResponseForbidden
from uploads.models import EventLog


@login_required
def preprocessing_dashboard_view(request):
    """
    Main preprocessing dashboard (Disco-like UI).
    Shows list of event logs and preprocessing interface.
    """
    # Only analysts and admins can access
    if not (request.user.roles.filter(name='Analyst').exists() or 
            request.user.roles.filter(name='Admin').exists()):
        return HttpResponseForbidden('دسترسی ندارید.')
    
    # Get selected log ID from query param
    selected_log_id = request.GET.get('log_id')
    selected_log = None
    
    if selected_log_id:
        try:
            selected_log = EventLog.objects.select_related('uploaded_file').get(
                pk=selected_log_id
            )
        except EventLog.DoesNotExist:
            pass
    
    # Get all event logs for sidebar
    event_logs = EventLog.objects.select_related('uploaded_file').all()
    
    context = {
        'event_logs': event_logs,
        'selected_log': selected_log,
    }
    
    return render(request, 'preprocessing/dashboard.html', context)


@login_required
def smart_clean_view(request, log_id):
    """
    Smart Clean page for a specific event log.
    """
    import json
    from .services import get_event_log_dataframe, _compute_log_stats
    
    # Only analysts and admins can access
    if not (request.user.roles.filter(name='Analyst').exists() or 
            request.user.roles.filter(name='Admin').exists()):
        return HttpResponseForbidden('دسترسی ندارید.')
    
    event_log = get_object_or_404(
        EventLog.objects.select_related('uploaded_file'),
        pk=log_id
    )
    
    # Compute statistics for both RAW and CLEANED
    raw_stats = None
    cleaned_stats = None
    
    try:
        raw_df = get_event_log_dataframe(event_log, version='raw')
        raw_stats = _compute_log_stats(raw_df)
        print(f"✅ RAW stats computed: {raw_stats}")
    except Exception as e:
        print(f"❌ Error computing RAW stats: {e}")
    
    if event_log.has_cleaned_version:
        try:
            cleaned_df = get_event_log_dataframe(event_log, version='cleaned')
            cleaned_stats = _compute_log_stats(cleaned_df)
            print(f"✅ CLEANED stats computed: {cleaned_stats}")
        except Exception as e:
            print(f"❌ Error computing CLEANED stats: {e}")
    
    context = {
        'event_log': event_log,
        'raw_stats': raw_stats,
        'cleaned_stats': cleaned_stats,
        'raw_stats_json': json.dumps(raw_stats) if raw_stats else 'null',
        'cleaned_stats_json': json.dumps(cleaned_stats) if cleaned_stats else 'null',
    }
    
    return render(request, 'preprocessing/smart_clean.html', context)
