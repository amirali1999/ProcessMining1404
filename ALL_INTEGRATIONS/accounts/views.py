from functools import wraps
from django.contrib import messages
from django.contrib.auth import authenticate, login, logout, get_user_model
from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import AuthenticationForm
from django.http import HttpResponseForbidden
from django.shortcuts import redirect, render
from .forms import RegisterForm, LicenseActivationForm
from .models import Role

User = get_user_model()


def role_required(role_name: str):
    def decorator(view_func):
        @login_required
        @wraps(view_func)
        def _wrapped(request, *args, **kwargs):
            if request.user.roles.filter(name=role_name).exists():
                return view_func(request, *args, **kwargs)
            return HttpResponseForbidden('دسترسی مجاز نیست.')
        return _wrapped
    return decorator


def register_view(request):
    if request.user.is_authenticated:
        return redirect('dashboard')
    if request.method == 'POST':
        form = RegisterForm(request.POST)
        if form.is_valid():
            user = form.save(commit=False)
            user.email = form.cleaned_data['email'].lower()
            user.save()
            analyst_role, _ = Role.objects.get_or_create(name='Analyst')
            user.roles.add(analyst_role)
            login(request, user)
            messages.success(request, 'ثبت‌نام با موفقیت انجام شد.')
            return redirect('dashboard')
        else:
            # If form has errors, display them
            for field, errors in form.errors.items():
                for error in errors:
                    messages.error(request, f'{error}')
    else:
        form = RegisterForm()
    return render(request, 'accounts/register.html', {'form': form})


def login_view(request):
    if request.user.is_authenticated:
        return redirect('dashboard')
    
    form = AuthenticationForm(request, data=request.POST or None)
    if request.method == 'POST' and form.is_valid():
        user = form.get_user()
        login(request, user)
        return redirect('dashboard')
    
    return render(request, 'accounts/login.html', {'form': form})


def admin_login_view(request):
    """Separate login page for admin panel access"""
    if request.user.is_authenticated:
        if request.user.is_staff or request.user.is_superuser:
            return redirect('/admin/')
        return redirect('dashboard')
    
    form = AuthenticationForm(request, data=request.POST or None)
    if request.method == 'POST' and form.is_valid():
        user = form.get_user()
        
        # Only allow staff/superuser to login through admin login
        if user.is_staff or user.is_superuser:
            login(request, user)
            return redirect('/admin/')
        else:
            messages.error(request, 'شما دسترسی به پنل ادمین ندارید.')
            return render(request, 'accounts/admin_login.html', {'form': form})
    
    return render(request, 'accounts/admin_login.html', {'form': form})


@login_required
def logout_view(request):
    logout(request)
    return redirect('login')


@login_required
def dashboard_view(request, project_name=None):
    from discovery.models import EventLogJob
    from uploads.models import EventLog
    from django.db.models import Max
    
    # Get unique project names from EventLogJob
    job_project_names = EventLogJob.objects.filter(
        user=request.user
    ).exclude(
        project_name__isnull=True
    ).exclude(
        project_name__exact=''
    ).values('project_name').annotate(
        latest_created=Max('created_at')
    ).order_by('-latest_created')
    
    # Get project names from EventLog (CSV imports)
    event_logs = EventLog.objects.filter(
        uploaded_file__uploader=request.user
    ).select_related('uploaded_file').order_by('-created_at')
    
    # Combine both sources - use a dict to avoid duplicates and keep the latest
    projects_dict = {}
    
    # Add EventLogJob projects
    for proj in job_project_names:
        projects_dict[proj['project_name']] = proj['latest_created']
    
    # Add EventLog projects
    for log in event_logs:
        if log.name not in projects_dict or log.created_at > projects_dict[log.name]:
            projects_dict[log.name] = log.created_at    # Sort by date (most recent first) and extract names
    projects = sorted(projects_dict.items(), key=lambda x: x[1], reverse=True)
    projects = [proj[0] for proj in projects]
    
    # Calculate usage statistics for free users (count unique project names)
    current_project_count = len(projects)
    
    # If project_name is specified, get the latest completed job for that project
    # Otherwise, get the latest completed job for any project
    if project_name:
        latest_job = EventLogJob.objects.filter(
            user=request.user,
            status='done',
            project_name=project_name
        ).order_by('-created_at').first()
        selected_project = project_name
    else:
        latest_job = EventLogJob.objects.filter(
            user=request.user,
            status='done'
        ).order_by('-created_at').first()
        selected_project = latest_job.project_name if latest_job else None
    
    return render(request, 'dashboard.html', {
        'latest_job': latest_job,
        'projects': projects,
        'selected_project': selected_project,
        'current_project_count': current_project_count,
    })


@role_required('Admin')
def admin_only_view(request):
    return render(request, 'admin_only.html')


@login_required
def projects_view(request):
    """
    Display the Disco-like Project Browser page with 3-column layout:
    - Left: Datasets list
    - Center: Map preview workspace
    - Right: Details panel
    """
    from discovery.models import EventLogJob
    from django.db.models import Max
    
    # Get unique project names with their latest job info
    # Group by project_name and get the latest job for each project
    project_names = EventLogJob.objects.filter(
        user=request.user
    ).exclude(
        project_name__isnull=True
    ).exclude(
        project_name__exact=''
    ).values('project_name').annotate(
        latest_created=Max('created_at')    ).order_by('-latest_created')
    
    # Build a list of projects with their details
    projects = []
    for proj in project_names:
        latest_job = EventLogJob.objects.filter(
            user=request.user,
            project_name=proj['project_name'],
            created_at=proj['latest_created']
        ).first()
        
        if latest_job:
            # Get the process map URL (prefer SVG over PNG)
            map_url = None
            if latest_job.output_map_svg:
                map_url = latest_job.output_map_svg.url
            elif latest_job.output_map_image:
                map_url = latest_job.output_map_image.url
            
            # Debug logging
            print(f"🔍 Project: {proj['project_name']}")
            print(f"   Job ID: {latest_job.id}")
            print(f"   Status: {latest_job.status}")
            print(f"   Has SVG: {bool(latest_job.output_map_svg)}")
            print(f"   Has PNG: {bool(latest_job.output_map_image)}")
            print(f"   Map URL: {map_url}")
            
            projects.append({
                'name': proj['project_name'],
                'created_at': latest_job.created_at,
                'job': latest_job,
                'map_url': map_url
            })
    
    return render(request, 'accounts/projects.html', {
        'projects': projects
    })


@login_required
def delete_project(request, project_name):
    """
    Delete all jobs associated with a project name.
    This removes the project from the database and all its related files.
    """
    from discovery.models import EventLogJob
    from django.http import JsonResponse
    import os
    from django.conf import settings
    
    if request.method != 'POST':
        return JsonResponse({'error': 'Only POST method is allowed'}, status=405)
    
    try:
        # Find all jobs with this project name for the current user
        jobs = EventLogJob.objects.filter(
            user=request.user,
            project_name=project_name
        )
        
        if not jobs.exists():
            return JsonResponse({'error': 'Project not found'}, status=404)
        
        # Count jobs to be deleted
        job_count = jobs.count()
        
        # Delete associated files
        for job in jobs:
            # Delete original file
            if job.original_file:
                try:
                    if os.path.isfile(job.original_file.path):
                        os.remove(job.original_file.path)
                except Exception as e:
                    print(f"Error deleting original file: {e}")
            
            # Delete output files (PNML and maps)
            if hasattr(job, 'output_pnml') and job.output_pnml:
                try:
                    if os.path.isfile(job.output_pnml.path):
                        os.remove(job.output_pnml.path)
                except Exception as e:
                    print(f"Error deleting PNML file: {e}")
            
            if job.output_map_image:
                try:
                    if os.path.isfile(job.output_map_image.path):
                        os.remove(job.output_map_image.path)
                except Exception as e:
                    print(f"Error deleting map image: {e}")
                    
            if hasattr(job, 'output_map_svg') and job.output_map_svg:
                try:
                    if os.path.isfile(job.output_map_svg.path):
                        os.remove(job.output_map_svg.path)
                except Exception as e:
                    print(f"Error deleting SVG map: {e}")
        
        # Delete all jobs from database
        jobs.delete()
        
        return JsonResponse({
            'success': True,
            'message': f'Project "{project_name}" deleted successfully',
            'deleted_jobs': job_count
        })
        
    except Exception as e:
        return JsonResponse({
            'error': f'Failed to delete project: {str(e)}'
        }, status=500)


@login_required
def export_project(request, project_name):
    """
    Export a project's event log file.
    Returns the original uploaded file for download.
    """
    from discovery.models import EventLogJob
    from django.http import JsonResponse, FileResponse, HttpResponse
    from urllib.parse import unquote
    import os
    
    # Decode URL-encoded project name
    project_name = unquote(project_name)
    
    try:
        # Find the latest job for this project
        job = EventLogJob.objects.filter(
            user=request.user,
            project_name=project_name
        ).order_by('-created_at').first()
        
        if not job:
            return JsonResponse({'error': 'Project not found'}, status=404)
        
        # Get the file to export
        if job.original_file:
            file_path = job.original_file.path
            if os.path.exists(file_path):
                # Determine content type
                filename = job.original_filename
                if filename.endswith('.xes'):
                    content_type = 'application/xml'
                elif filename.endswith('.csv'):
                    content_type = 'text/csv'
                else:
                    content_type = 'application/octet-stream'
                
                # Open and return the file
                response = FileResponse(
                    open(file_path, 'rb'),
                    content_type=content_type
                )
                response['Content-Disposition'] = f'attachment; filename="{filename}"'
                return response
        
        return JsonResponse({'error': 'File not found'}, status=404)
        
    except Exception as e:
        import traceback
        print(f"Export error: {str(e)}")
        print(traceback.format_exc())
        return JsonResponse({'error': str(e)}, status=500)


@login_required
def get_project_details(request, project_name):
    """
    API endpoint to get project details by name.
    Returns the latest job/event log information for the specified project.
    """
    from discovery.models import EventLogJob
    from uploads.models import EventLog
    from django.http import JsonResponse
    from urllib.parse import unquote
    
    # Decode URL-encoded project name
    project_name = unquote(project_name)
    
    try:
        # First, try to find an EventLogJob with this project name
        job = EventLogJob.objects.filter(
            user=request.user,
            project_name=project_name
        ).order_by('-created_at').first()
        
        if job:
            # Return EventLogJob data
            return JsonResponse({
                'project_name': job.project_name,
                'original_filename': job.original_filename,
                'created_at': job.created_at.isoformat(),
                'updated_at': job.updated_at.isoformat(),
                'status': job.status,
                'mining_method': job.mining_method,
                'mining_method_display': job.get_mining_method_display(),
                'cleaning_enabled': job.cleaning_enabled,
                'output_map_image': job.output_map_image.url if job.output_map_image else None,
                'output_map_svg': job.output_map_svg.url if job.output_map_svg else None,
                'source': 'job'
            })
        
        # If no job found, try EventLog
        event_log = EventLog.objects.filter(
            uploaded_file__uploader=request.user,
            name=project_name
        ).select_related('uploaded_file').order_by('-created_at').first()
        
        if event_log:
            # Return EventLog data
            return JsonResponse({
                'project_name': event_log.name,
                'original_filename': event_log.uploaded_file.original_name,
                'created_at': event_log.created_at.isoformat(),
                'updated_at': event_log.updated_at.isoformat(),
                'status': 'completed',  # EventLogs are always completed
                'file_type': event_log.file_type,
                'num_cases': event_log.num_cases,
                'num_events': event_log.num_events,
                'num_activities': event_log.num_activities,
                'has_cleaned_version': event_log.has_cleaned_version,
                'output_map_image': None,  # EventLog doesn't have map yet
                'source': 'eventlog'
            })
        
        # Project not found
        return JsonResponse({
            'error': 'Project not found'
        }, status=404)
        
    except Exception as e:
        import traceback
        print(f"Error fetching project details: {str(e)}")
        print(traceback.format_exc())
        return JsonResponse({
            'error': str(e)
        }, status=500)


@login_required
def activate_license_view(request):
    """
    View for activating premium license with code
    """
    from .forms import LicenseActivationForm
    
    # If user is already premium, redirect to dashboard
    if request.user.is_premium:
        messages.info(request, 'شما در حال حاضر کاربر پرمیوم هستید.')
        return redirect('dashboard')
    
    if request.method == 'POST':
        form = LicenseActivationForm(request.POST, user=request.user)
        if form.is_valid():
            # Activate license
            if form.activate():
                messages.success(
                    request,
                    '🎉 تبریک! لایسنس پرمیوم شما با موفقیت فعال شد. '
                    'اکنون از تمام امکانات بدون محدودیت استفاده کنید.'
                )
                return redirect('dashboard')
            else:
                messages.error(request, 'خطا در فعال‌سازی لایسنس. لطفاً دوباره تلاش کنید.')
    else:
        form = LicenseActivationForm(user=request.user)
    
    return render(request, 'accounts/activate_license.html', {
        'form': form,
        'user_limitations': {
            'max_log_rows': request.user.max_log_rows,
            'max_projects': request.user.max_projects,
            'allowed_algorithms': request.user.get_allowed_algorithms(),
        }
    })
