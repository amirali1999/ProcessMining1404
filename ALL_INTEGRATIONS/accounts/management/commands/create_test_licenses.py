"""
Management command to create initial test license codes
Usage: python manage.py create_test_licenses
"""
from django.core.management.base import BaseCommand
from accounts.models import LicenseCode, User


class Command(BaseCommand):
    help = 'Create initial test license codes for development'

    def handle(self, *args, **options):
        # Get or create admin user as creator
        admin_user = User.objects.filter(is_superuser=True).first()
        
        # Define test codes
        test_codes = [
            ('PREMIUM-2024', 0, 'Lifetime premium - Test code 2024'),
            ('PREMIUM-TEST', 0, 'Lifetime premium - General test'),
            ('PREMIUM-30DAY', 30, '30-day premium trial'),
            ('PREMIUM-DEMO', 7, '7-day premium demo'),
        ]
        
        self.stdout.write('\n' + '='*60)
        self.stdout.write(self.style.SUCCESS('\n🎫 Creating Test License Codes...\n'))
        self.stdout.write('='*60 + '\n')
        
        created = 0
        skipped = 0
        
        for code, duration, description in test_codes:
            # Check if code already exists
            if LicenseCode.objects.filter(code=code).exists():
                self.stdout.write(
                    self.style.WARNING(f'⚠️  Skipped: {code} (already exists)')
                )
                skipped += 1
                continue
            
            # Create license code
            LicenseCode.objects.create(
                code=code,
                duration_days=duration,
                created_by=admin_user,
                is_active=True
            )
            
            duration_text = 'Lifetime' if duration == 0 else f'{duration} days'
            self.stdout.write(
                self.style.SUCCESS(f'✓ Created: {code} ({duration_text})')
            )
            self.stdout.write(f'   📝 {description}')
            created += 1
        
        self.stdout.write('\n' + '='*60)
        self.stdout.write(
            self.style.SUCCESS(f'\n✅ Created {created} new codes, skipped {skipped} existing\n')
        )
        self.stdout.write('='*60 + '\n')
        
        if created > 0:
            self.stdout.write('\n🔑 TEST CODES:\n')
            for code, duration, description in test_codes:
                if LicenseCode.objects.filter(code=code, is_used=False).exists():
                    self.stdout.write(f'   • {code}')
            
            self.stdout.write('\n' + '='*60)
            self.stdout.write('\n💡 TIP: Use these codes to test license activation at:')
            self.stdout.write('   Registration: /accounts/register/')
            self.stdout.write('   Activation: /accounts/activate-license/')
            self.stdout.write('\n' + '='*60 + '\n')
