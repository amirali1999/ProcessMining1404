from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views, web_views

# API Routes (REST Framework)
# Discovery API endpoints are mounted under event-logs for cleaner nesting
api_urlpatterns = [
    # Discovery endpoints nested under event logs
    path('event-logs/<int:pk>/discovery/alpha/', views.DiscoveryViewSet.as_view({'post': 'run_alpha'}), name='discovery-run-alpha'),
    path('event-logs/<int:pk>/discovery/heuristics/', views.DiscoveryViewSet.as_view({'post': 'run_heuristics'}), name='discovery-run-heuristics'),
    path('event-logs/<int:pk>/discovery/models/', views.DiscoveryViewSet.as_view({'get': 'list_models'}), name='discovery-list-models'),
    
    # Direct model access
    path('discovered-models/<int:pk>/pnml/', views.DiscoveredModelViewSet.as_view({'get': 'get_pnml'}), name='discovered-model-pnml'),
    # Group 5: Visualization endpoints
    path('discovered-models/<int:pk>/petrinet-image/', views.DiscoveredModelViewSet.as_view({'get': 'petrinet_image'}), name='discovered-model-petrinet-image'),
    path('discovered-models/<int:pk>/petrinet-svg/', views.DiscoveredModelViewSet.as_view({'get': 'petrinet_svg'}), name='discovered-model-petrinet-svg'),
    
    # Project API for dashboard dropdown
    path('projects/<str:project_name>/', views.get_project_api, name='get-project-api'),
]

# Web Routes
web_urlpatterns = [
    path('discovery/', web_views.discovery_dashboard, name='discovery_dashboard'),
    path('discovery/<int:event_log_id>/discover/', web_views.discover_view, name='discover_view'),
    path('visualize/<int:model_id>/', web_views.visualize_view, name='visualize_model'),
]

# Job Processing Routes
job_urlpatterns = [
    path('jobs/create/', views.create_job, name='job_create'),
    path('jobs/status/<int:job_id>/', views.job_status, name='job_status'),
    path('jobs/progress/<int:job_id>/', views.job_progress_page, name='job_progress'),
]

# CSV Import Routes
csv_import_urlpatterns = [
    path('import/csv/create/', views.create_csv_import_session, name='csv_import_create'),
    path('import/csv/<int:session_id>/', views.csv_import_page, name='csv_import'),
    path('import/csv/<int:session_id>/process/', views.process_csv_import, name='csv_import_process'),
]

urlpatterns = api_urlpatterns + web_urlpatterns + job_urlpatterns + csv_import_urlpatterns
