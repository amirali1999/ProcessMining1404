from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
from django.db.models import Sum
from .forms import EventLogUploadForm
from .models import EventLog

def upload_event_log(request):

    if request.method == 'POST':
        form = EventLogUploadForm(request.POST, request.FILES)
        if form.is_valid():
            try:
                event_log = form.save(
                    user=request.user if request.user.is_authenticated else None
                )
                
                messages.success(
                    request,
                    f'✅ فایل "{event_log.original_filename}" با موفقیت آپلود شد.'
                )
                
                return redirect(f'/upload/success/{event_log.pk}/')
            
            except Exception as e:
                messages.error(
                    request,
                    f'❌ خطا در آپلود فایل: {str(e)}'
                )
        else:
            for field, errors in form.errors.items():
                for error in errors:
                    messages.error(request, error)
    else:
        form = EventLogUploadForm()
    
    recent_logs = EventLog.objects.all()[:5]
    
    total_count = EventLog.objects.count()
    total_size = EventLog.objects.aggregate(Sum('file_size'))['file_size__sum'] or 0
    total_size_mb = round(total_size / (1024 * 1024), 2)
    
    context = {
        'form': form,
        'recent_logs': recent_logs,
        'total_count': total_count,
        'total_size_mb': total_size_mb,
    }
    
    return render(request, 'upload_app/upload.html', context)


def upload_success(request, pk):

    event_log = get_object_or_404(EventLog, pk=pk)
    
    context = {
        'event_log': event_log,
    }
    
    return render(request, 'upload_app/success.html', context)


def event_log_list(request):

    logs = EventLog.objects.all()
    
    total_count = logs.count()
    total_size = logs.aggregate(Sum('file_size'))['file_size__sum'] or 0
    total_size_mb = round(total_size / (1024 * 1024), 2)
    
    context = {
        'logs': logs,
        'total_count': total_count,
        'total_size_mb': total_size_mb,
    }
    
    return render(request, 'upload_app/list.html', context)


def event_log_detail(request, pk):

    event_log = get_object_or_404(EventLog, pk=pk)
    
    context = {
        'event_log': event_log,
    }
    
    return render(request, 'upload_app/detail.html', context)
