
from django.urls import path
from . import views

app_name = "monitoring"

urlpatterns = [
    path("", views.index, name="index"),
    path("upload/", views.upload_and_run, name="upload"),
    path("runs/<int:run_id>/", views.run_detail, name="run_detail"),
    path("runs/<int:run_id>/delete/", views.run_delete, name="run_delete"),
    path("runs/<int:run_id>/download/<str:kind>/", views.download_run_file, name="run_download"),

]
