from django.contrib import admin
from .models import EventLog

@admin.register(EventLog)
class EventLogAdmin(admin.ModelAdmin):
    list_display = [
        'id',
        'original_filename',
        'file_size_display',
        'uploaded_at',
        'uploaded_by'
    ]
    list_filter = ['uploaded_at']
    search_fields = ['original_filename', 'file']
    readonly_fields = ['uploaded_at', 'file_size', 'file']
    date_hierarchy = 'uploaded_at'
    
    def file_size_display(self, obj):
        return f"{obj.get_file_size_mb()} MB"
    file_size_display.short_description = 'حجم فایل'
    
    def has_add_permission(self, request):
        return False
