from django.shortcuts import redirect
from django.http import HttpRequest, HttpResponse


def set_language(request: HttpRequest) -> HttpResponse:
    """
    View to set language preference in session
    """
    language = request.GET.get('lang', 'fa')
    if language in ['fa', 'en']:
        request.session['language'] = language
    
    # Redirect back to the page user came from
    return redirect(request.META.get('HTTP_REFERER', '/'))
