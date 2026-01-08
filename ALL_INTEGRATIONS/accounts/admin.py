from django.contrib import admin
from django.contrib.auth.admin import UserAdmin as DjangoUserAdmin
from django.contrib.auth.models import Group
from django.utils.html import format_html
from django.utils.safestring import mark_safe
from django.utils import timezone
from rest_framework.authtoken.models import Token
from rest_framework.authtoken.admin import TokenAdmin
from .models import User, Role, LicenseCode

# Customize admin site titles
admin.site.site_header = "Process Mining Administration"
admin.site.site_title = "Process Mining Admin"
admin.site.index_title = "Dashboard"


@admin.register(User)
class UserAdmin(DjangoUserAdmin):
    change_list_template = 'admin/accounts/user/change_list.html'
    
    fieldsets = DjangoUserAdmin.fieldsets + (
        ('Roles', {'fields': ('roles',)}),
        ('License Information', {
            'fields': (
                'license_type',
                'license_activated_at',
                'license_expires_at',
            ),
            'classes': ('collapse',),
        }),        ('Usage Limitations', {
            'fields': (
                'max_log_rows',
                'allowed_algorithms',
                'max_projects',
            ),
            'classes': ('collapse',),
            'description': 'Set limitations for FREE users (0 = unlimited, empty list = all allowed)'
        }),
    )
    
    list_display = (
        'username',
        'email',
        'license_badge',
        'license_status',
        'max_log_rows',
        'max_projects',
        'is_staff',
        'date_joined'
    )
    
    list_filter = (
        'license_type',
        'is_staff',
        'is_active',
        'date_joined'
    )
    
    search_fields = ('username', 'email', 'first_name', 'last_name')
    filter_horizontal = ('roles',)
    
    actions = ['activate_premium_lifetime', 'activate_premium_30days', 'downgrade_to_free']
    
    def has_add_permission(self, request):
        """Ensure superusers can always add users"""
        return request.user.is_superuser or super().has_add_permission(request)
    
    def license_badge(self, obj):
        """Display license type with color badge"""
        if obj.is_premium:
            return mark_safe(
                '<span style="background-color: #28a745; color: white; padding: 3px 8px; border-radius: 3px; font-weight: bold;">👑 PREMIUM</span>'
            )
        return mark_safe(
            '<span style="background-color: #6c757d; color: white; padding: 3px 8px; border-radius: 3px;">FREE</span>'
        )
    license_badge.short_description = 'License'
    
    def license_status(self, obj):
        """Display license expiration status"""
        if not obj.is_premium:
            return '-'
        
        if not obj.license_expires_at:
            return mark_safe('<span style="color: green;">✓ Lifetime</span>')
        
        days_left = (obj.license_expires_at - timezone.now()).days
        if days_left < 0:
            return mark_safe('<span style="color: red;">✗ Expired</span>')
        elif days_left < 7:
            return format_html(
                '<span style="color: orange;">⚠ {} days left</span>',
                days_left
            )
        else:
            return format_html(
                '<span style="color: green;">✓ {} days left</span>',
                days_left
            )
    license_status.short_description = 'Status'
    
    def activate_premium_lifetime(self, request, queryset):
        """Activate premium license (lifetime) for selected users"""
        count = 0
        for user in queryset:
            user.activate_premium_license(duration_days=None)
            count += 1
        self.message_user(request, f'{count} user(s) upgraded to Premium (Lifetime)')
    activate_premium_lifetime.short_description = '👑 Activate Premium (Lifetime)'
    
    def activate_premium_30days(self, request, queryset):
        """Activate premium license (30 days) for selected users"""
        count = 0
        for user in queryset:
            user.activate_premium_license(duration_days=30)
            count += 1
        self.message_user(request, f'{count} user(s) upgraded to Premium (30 days)')
    activate_premium_30days.short_description = '👑 Activate Premium (30 days)'
    
    def downgrade_to_free(self, request, queryset):
        """Downgrade selected users to free plan"""
        count = queryset.update(
            license_type=User.LICENSE_FREE,
            max_log_rows=1000,
            max_projects=3,
            allowed_algorithms=['alpha']  # Only Alpha miner for free users
        )
        self.message_user(request, f'{count} user(s) downgraded to Free')
    downgrade_to_free.short_description = '📉 Downgrade to Free'


@admin.register(Role)
class RoleAdmin(admin.ModelAdmin):
    change_list_template = 'admin/accounts/role/change_list.html'
    
    list_display = ('id', 'name')
    search_fields = ('name',)
    
    def has_add_permission(self, request):
        """Ensure superusers can always add roles"""
        return request.user.is_superuser or super().has_add_permission(request)


@admin.register(LicenseCode)
class LicenseCodeAdmin(admin.ModelAdmin):
    change_list_template = 'admin/accounts/licensecode/change_list.html'
    
    list_display = (
        'code',
        'status_badge',
        'duration_display',
        'used_by',
        'used_at',
        'created_at'
    )
    
    list_filter = ('is_active', 'is_used', 'created_at')
    search_fields = ('code', 'used_by__username', 'notes')
    readonly_fields = ('is_used', 'used_by', 'used_at', 'created_at', 'created_by')
    
    fieldsets = (
        ('Code Information', {
            'fields': ('code', 'is_active', 'duration_days', 'notes')
        }),
        ('Usage Information', {
            'fields': ('is_used', 'used_by', 'used_at'),
            'classes': ('collapse',)
        }),
        ('Metadata', {
            'fields': ('created_at', 'created_by'),
            'classes': ('collapse',)
        }),
    )
    
    actions = ['deactivate_codes', 'activate_codes']
    
    def status_badge(self, obj):
        """Display code status with badge"""
        if obj.is_used:
            return mark_safe(
                '<span style="background-color: #dc3545; color: white; padding: 3px 8px; border-radius: 3px;">USED</span>'
            )
        elif obj.is_active:
            return mark_safe(
                '<span style="background-color: #28a745; color: white; padding: 3px 8px; border-radius: 3px;">ACTIVE</span>'
            )
        else:
            return mark_safe(
                '<span style="background-color: #6c757d; color: white; padding: 3px 8px; border-radius: 3px;">INACTIVE</span>'
            )
    status_badge.short_description = 'Status'
    
    def duration_display(self, obj):
        """Display license duration"""
        if obj.duration_days is None:
            return '∞ Lifetime'
        return f'{obj.duration_days} days'
    duration_display.short_description = 'Duration'
    
    def save_model(self, request, obj, form, change):
        """Auto-set created_by on creation"""
        if not change:  # New object
            obj.created_by = request.user
        super().save_model(request, obj, form, change)
    
    def deactivate_codes(self, request, queryset):
        """Deactivate selected codes"""
        count = queryset.filter(is_used=False).update(is_active=False)
        self.message_user(request, f'{count} code(s) deactivated')
    deactivate_codes.short_description = '🚫 Deactivate selected codes'
    
    def activate_codes(self, request, queryset):
        """Activate selected codes"""
        count = queryset.filter(is_used=False).update(is_active=True)
        self.message_user(request, f'{count} code(s) activated')
    activate_codes.short_description = '✅ Activate selected codes'


# ==================== TOKEN PROXY ADMIN ====================
# Try to unregister the default Token admin if it exists
try:
    admin.site.unregister(Token)
except admin.sites.NotRegistered:
    pass  # Token was not registered, so we can proceed

# Create a proxy model to customize the admin display name
class TokenProxy(Token):
    class Meta:
        proxy = True
        verbose_name = 'API Token'
        verbose_name_plural = 'API Tokens'


@admin.register(TokenProxy)
class TokenProxyAdmin(TokenAdmin):
    change_list_template = 'admin/authtoken/tokenproxy/change_list.html'
    
    list_display = ('key', 'user', 'created')
    fields = ('user',)
    ordering = ('-created',)
    
    def has_add_permission(self, request):
        """Ensure superusers can always add tokens"""
        return request.user.is_superuser or super().has_add_permission(request)
    
    def changelist_view(self, request, extra_context=None):
        """Add list of users to context for inline form"""
        extra_context = extra_context or {}
        extra_context['all_users'] = User.objects.all().order_by('username')
        return super().changelist_view(request, extra_context=extra_context)
