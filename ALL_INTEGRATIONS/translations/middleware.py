"""
Middleware to ensure language is always set in session
"""


class LanguageMiddleware:
    """
    Ensures that every request has a language set in the session.
    If no language is set, defaults to 'fa' (Persian).
    """
    
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        # Set default language if not already set
        if 'language' not in request.session:
            request.session['language'] = 'fa'
        
        response = self.get_response(request)
        return response
