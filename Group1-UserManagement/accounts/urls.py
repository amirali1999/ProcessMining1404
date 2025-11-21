from django.urls import path
from .views import register_view, login_view, logout_view, dashboard_view, admin_only_view

urlpatterns = [
    path('', dashboard_view, name='dashboard'),
    path('register/', register_view, name='register'),
    path('login/', login_view, name='login'),
    path('logout/', logout_view, name='logout'),
    path('admin-only/', admin_only_view, name='admin_only'),
]
