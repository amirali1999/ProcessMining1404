from django.contrib import admin
from .models import Translation


@admin.register(Translation)
class TranslationAdmin(admin.ModelAdmin):
    change_list_template = 'admin/translations/translation/change_list.html'
    
    list_display = ['phrase', 'fa', 'en']
    search_fields = ['phrase', 'fa', 'en']
    list_filter = []
    ordering = ['phrase']
    
    def has_add_permission(self, request):
        """Ensure superusers can always add translations"""
        return request.user.is_superuser or super().has_add_permission(request)
