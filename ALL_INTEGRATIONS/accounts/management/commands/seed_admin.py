from django.core.management.base import BaseCommand
from django.contrib.auth import get_user_model
from accounts.models import Role
import os


class Command(BaseCommand):
    help = "Create a default superuser 'admin' with password from env DEFAULT_ADMIN_PASSWORD (fallback: Admin@12345) and assign Admin role."

    def add_arguments(self, parser):
        parser.add_argument('--username', default=os.environ.get('DEFAULT_ADMIN_USERNAME', 'admin'))
        parser.add_argument('--email', default=os.environ.get('DEFAULT_ADMIN_EMAIL', 'admin@example.com'))
        parser.add_argument('--password', default=os.environ.get('DEFAULT_ADMIN_PASSWORD', 'Admin@12345'))
        parser.add_argument('--reset', action='store_true', help='Reset password if user exists')

    def handle(self, *args, **options):
        User = get_user_model()
        username = options['username']
        email = options['email']
        password = options['password']
        reset = options['reset']

        user, created = User.objects.get_or_create(username=username, defaults={
            'email': email,
            'is_staff': True,
            'is_superuser': True,
        })

        if created:
            user.set_password(password)
            user.save()
            self.stdout.write(self.style.SUCCESS(f"Created superuser '{username}'"))
        else:
            updated = False
            if reset:
                user.set_password(password)
                updated = True
            if not user.is_staff or not user.is_superuser:
                user.is_staff = True
                user.is_superuser = True
                updated = True
            if updated:
                user.save()
                self.stdout.write(self.style.WARNING(f"Updated existing user '{username}' (permissions/password)"))
            else:
                self.stdout.write(self.style.NOTICE(f"User '{username}' already exists; no changes."))

        # Ensure Admin role exists and is assigned
        admin_role, _ = Role.objects.get_or_create(name='Admin')
        if not user.roles.filter(name='Admin').exists():
            user.roles.add(admin_role)
            self.stdout.write(self.style.SUCCESS("Assigned 'Admin' role to the user"))
        else:
            self.stdout.write("'Admin' role already assigned")

        self.stdout.write(self.style.SUCCESS("Admin seeding completed."))
