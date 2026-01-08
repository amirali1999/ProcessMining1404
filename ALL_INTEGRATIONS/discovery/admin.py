from django.contrib import admin
from .models import DiscoveredProcessModel, EventLogJob


@admin.register(DiscoveredProcessModel)
class DiscoveredProcessModelAdmin(admin.ModelAdmin):
    list_display = ['id', 'event_log', 'algorithm', 'source_version', 'num_places', 'num_transitions', 'discovered_at']
    list_filter = ['algorithm', 'source_version', 'discovered_at']
    search_fields = ['event_log__name']
    readonly_fields = ['discovered_at', 'pnml_content', 'num_places', 'num_transitions', 'num_arcs']
    
    fieldsets = (
        ('Basic Information', {
            'fields': ('event_log', 'algorithm', 'source_version', 'discovered_by')
        }),
        ('Model Statistics', {
            'fields': ('num_places', 'num_transitions', 'num_arcs')
        }),
        ('PNML Data', {
            'fields': ('pnml_content',),
            'classes': ('collapse',)
        }),
        ('Metadata', {
            'fields': ('discovered_at',)
        }),
    )


@admin.register(EventLogJob)
class EventLogJobAdmin(admin.ModelAdmin):
    list_display = ['id', 'user', 'original_filename', 'status', 'progress', 'mining_method', 'cleaning_enabled', 'created_at']
    list_filter = ['status', 'mining_method', 'cleaning_enabled', 'created_at']
    search_fields = ['original_filename', 'user__username']
    readonly_fields = ['created_at', 'updated_at', 'user', 'original_file', 'original_filename']
    
    fieldsets = (
        ('User & File', {
            'fields': ('user', 'original_file', 'original_filename')
        }),
        ('Processing Options', {
            'fields': ('cleaning_enabled', 'mining_method')
        }),
        ('Job Status', {
            'fields': ('status', 'progress', 'message', 'error_message')
        }),
        ('Output', {
            'fields': ('output_map_image', 'output_map_svg')
        }),
        ('Timestamps', {
            'fields': ('created_at', 'updated_at')
        }),
    )
    
    def has_add_permission(self, request):
        # Prevent manual creation in admin
        return False
