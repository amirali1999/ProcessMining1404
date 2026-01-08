from django.db.models.signals import post_migrate
from django.dispatch import receiver
from .models import Role


@receiver(post_migrate)
def create_default_roles(sender, **kwargs):
    # Only act when the accounts app is migrated
    if getattr(sender, 'name', None) != 'accounts':
        return
    Role.objects.get_or_create(name='Admin')
    Role.objects.get_or_create(name='Analyst')
