"""
Decorators for enforcing license limitations
"""
from functools import wraps
from django.shortcuts import redirect
from django.contrib import messages
from django.http import JsonResponse


def premium_required(view_func):
    """
    Decorator to require premium license for a view
    """
    @wraps(view_func)
    def _wrapped_view(request, *args, **kwargs):
        if not request.user.is_authenticated:
            return redirect('login')
        
        if not request.user.is_premium:
            messages.error(request, '⚠️ This feature requires a Premium license. Please upgrade your account.')
            return redirect('activate_license')
        
        return view_func(request, *args, **kwargs)
    return _wrapped_view


def check_max_projects(view_func):
    """
    Decorator to check if user has reached max projects limit
    """
    @wraps(view_func)
    def _wrapped_view(request, *args, **kwargs):
        if not request.user.is_authenticated:
            return redirect('login')
        
        # Premium users have unlimited projects
        if request.user.is_premium:
            return view_func(request, *args, **kwargs)
        
        # Check project count for free users
        max_projects = request.user.max_projects
        if max_projects > 0:  # 0 means unlimited
            from discovery.models import EventLogJob
            # Count unique project names for this user
            current_count = EventLogJob.objects.filter(
                user=request.user
            ).exclude(
                project_name__isnull=True
            ).exclude(
                project_name__exact=''
            ).values('project_name').distinct().count()
            
            if current_count >= max_projects:
                messages.error(
                    request,
                    f'⚠️ You have reached your project limit ({max_projects} projects). '
                    'Please upgrade to Premium for unlimited projects or delete an existing project.'
                )
                return redirect('activate_license')
        
        return view_func(request, *args, **kwargs)
    return _wrapped_view


def check_algorithm_access(view_func):
    """
    Decorator to check if user has access to requested algorithm
    """
    @wraps(view_func)
    def _wrapped_view(request, *args, **kwargs):
        if not request.user.is_authenticated:
            return redirect('login')
        
        # Get algorithm from request (POST or GET)
        algorithm = request.POST.get('algorithm') or request.GET.get('algorithm')
        
        if algorithm and not request.user.can_use_algorithm(algorithm):
            allowed = ', '.join([alg.title() for alg in request.user.get_allowed_algorithms()])
            messages.error(
                request,
                f'⚠️ You do not have access to the {algorithm.title()} algorithm. '
                f'Your plan allows: {allowed}. Please upgrade to Premium for access to all algorithms.'
            )
            return redirect('activate_license')
        
        return view_func(request, *args, **kwargs)
    return _wrapped_view


def check_log_size_limit(max_rows_param='max_rows'):
    """
    Decorator factory to check if uploaded log exceeds size limit
    Usage: @check_log_size_limit('row_count_field_name')
    """
    def decorator(view_func):
        @wraps(view_func)
        def _wrapped_view(request, *args, **kwargs):
            if not request.user.is_authenticated:
                return redirect('login')
            
            # Premium users have unlimited log size
            if request.user.is_premium:
                return view_func(request, *args, **kwargs)
            
            # Check log size for free users
            max_rows = request.user.max_log_rows
            if max_rows > 0:  # 0 means unlimited
                # This will be checked in the view after file is parsed
                # We'll pass the limit to the view via request
                request.user_max_log_rows = max_rows
            
            return view_func(request, *args, **kwargs)
        return _wrapped_view
    return decorator


def api_premium_required(view_func):
    """
    Decorator for API views that require premium
    Returns JSON response instead of redirect
    """
    @wraps(view_func)
    def _wrapped_view(request, *args, **kwargs):
        if not request.user.is_authenticated:
            return JsonResponse({
                'error': 'Authentication required'
            }, status=401)
        
        if not request.user.is_premium:
            return JsonResponse({
                'error': 'Premium license required',
                'message': 'This feature requires a Premium license. Please upgrade your account.',
                'upgrade_url': '/accounts/activate-license/'
            }, status=403)
        
        return view_func(request, *args, **kwargs)
    return _wrapped_view
