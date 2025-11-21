from django.urls import path
from .views import uploads_list_view, upload_file_view, download_file_view, delete_file_view

urlpatterns = [
    path('', uploads_list_view, name='uploads_list'),
    path('upload/', upload_file_view, name='upload_file'),
    path('<int:pk>/download/', download_file_view, name='download_file'),
    path('<int:pk>/delete/', delete_file_view, name='delete_file'),
]
