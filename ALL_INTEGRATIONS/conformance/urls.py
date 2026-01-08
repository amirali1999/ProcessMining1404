"""
Conformance Checking URL Configuration (Group 6)
"""
from django.urls import path
from . import views, web_views

# API Routes
api_urlpatterns = [
    # Run conformance check
    path('conformance/run/', views.ConformanceViewSet.as_view({'post': 'run_conformance'}), name='conformance-run'),
    
    # Get result summary
    path('conformance/results/<int:pk>/summary/', views.ConformanceViewSet.as_view({'get': 'get_result'}), name='conformance-result'),
    
    # Get cases (paginated)
    path('conformance/results/<int:pk>/cases/', views.ConformanceViewSet.as_view({'get': 'get_cases'}), name='conformance-cases'),
]

# Web Routes
web_urlpatterns = [
    path('conformance/<int:event_log_id>/', web_views.conformance_view, name='conformance_view'),
]

urlpatterns = api_urlpatterns + web_urlpatterns
