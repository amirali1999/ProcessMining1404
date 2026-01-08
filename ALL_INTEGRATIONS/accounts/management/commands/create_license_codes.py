"""
Management command to create premium license codes
Usage: python manage.py create_license_codes --count 10 --duration 365
"""
from django.core.management.base import BaseCommand
from accounts.models import LicenseCode, User
import random
import string


class Command(BaseCommand):
    help = 'Create premium license activation codes'

    def add_arguments(self, parser):
        parser.add_argument(
            '--count',
            type=int,
            default=5,
            help='Number of license codes to generate (default: 5)'
        )
        parser.add_argument(
            '--duration',
            type=int,
            default=0,
            help='Duration in days (0 = lifetime, default: 0)'
        )
        parser.add_argument(
            '--prefix',
            type=str,
            default='PREMIUM',
            help='Code prefix (default: PREMIUM)'
        )

    def handle(self, *args, **options):
        count = options['count']
        duration = options['duration']
        prefix = options['prefix']
        
        # Get or create admin user as creator
        admin_user = User.objects.filter(is_superuser=True).first()
        if not admin_user:
            self.stdout.write(
                self.style.WARNING('No admin user found. Creating codes without creator.')
            )
        
        created_codes = []
        
        for i in range(count):
            # Generate unique code
            while True:
                suffix = ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))
                code = f'{prefix}-{suffix}'
                
                # Check if code already exists
                if not LicenseCode.objects.filter(code=code).exists():
                    break
            
            # Create license code
            license_code = LicenseCode.objects.create(
                code=code,
                duration_days=duration,
                created_by=admin_user,
                is_active=True
            )
            
            created_codes.append(code)
            
            duration_text = 'Lifetime' if duration == 0 else f'{duration} days'
            self.stdout.write(
                self.style.SUCCESS(f'✓ Created: {code} ({duration_text})')
            )
        
        self.stdout.write('\n' + '='*60)
        self.stdout.write(self.style.SUCCESS(f'\n✅ Successfully created {count} license codes!\n'))
        self.stdout.write('='*60 + '\n')
        
        # Print summary
        self.stdout.write('\n📋 CODES SUMMARY:\n')
        for code in created_codes:
            self.stdout.write(f'   • {code}')
        
        self.stdout.write('\n' + '='*60)
        self.stdout.write('\n💡 TIP: You can activate these codes at:')
        self.stdout.write('   /accounts/activate-license/')
        self.stdout.write('\n' + '='*60 + '\n')
