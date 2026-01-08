from django.urls import path
from .views import (
    register_view, 
    login_view,
    admin_login_view,
    logout_view, 
    dashboard_view, 
    admin_only_view, 
    projects_view, 
    delete_project,
    get_project_details,
    export_project,
    activate_license_view
)

urlpatterns = [
    path('', dashboard_view, name='dashboard'),
    path('dashboard/', dashboard_view, name='dashboard_alt'),  # Alternative URL for dashboard
    path('dashboard/<path:project_name>/', dashboard_view, name='dashboard_project'),  # Dashboard for specific project
    path('projects/', projects_view, name='projects'),  # Disco-like Project Browser
    path('projects/delete/<str:project_name>/', delete_project, name='delete_project'),  # Delete project API
    path('projects/export/<path:project_name>/', export_project, name='export_project'),  # Export project API
    path('api/projects/<path:project_name>/', get_project_details, name='get_project_details'),  # Get project details API
    path('register/', register_view, name='register'),
    path('login/', login_view, name='login'),
    path('admin-login/', admin_login_view, name='admin_login'),
    path('logout/', logout_view, name='logout'),
    path('activate-license/', activate_license_view, name='activate_license'),  # Premium license activation
    path('admin-only/', admin_only_view, name='admin_only'),
]
