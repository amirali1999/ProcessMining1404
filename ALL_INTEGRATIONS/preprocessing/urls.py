from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import EventLogViewSet
from . import web_views

router = DefaultRouter()
router.register(r'event-logs', EventLogViewSet, basename='eventlog')

urlpatterns = [
    # REST API endpoints
    path('', include(router.urls)),
]

# Web UI URLs (separate from API)
web_urlpatterns = [
    path('preprocessing/', web_views.preprocessing_dashboard_view, name='preprocessing_dashboard'),
    path('preprocessing/<int:log_id>/smart-clean/', web_views.smart_clean_view, name='smart_clean'),
]
