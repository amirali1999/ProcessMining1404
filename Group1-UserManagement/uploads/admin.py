from django.contrib import admin
from .models import UploadedFile


@admin.register(UploadedFile)
class UploadedFileAdmin(admin.ModelAdmin):
    list_display = ("id", "original_name", "uploader", "uploaded_at", "size_kb")
    list_filter = ("uploaded_at", "uploader")
    search_fields = ("original_name", "description", "uploader__username")
