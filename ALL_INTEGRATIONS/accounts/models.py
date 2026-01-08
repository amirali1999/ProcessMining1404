from django.contrib.auth.models import AbstractUser
from django.db import models
from django.utils import timezone
from datetime import timedelta


class Role(models.Model):
    name = models.CharField(max_length=50, unique=True)

    def __str__(self) -> str:
        return self.name


class User(AbstractUser):
    # License Types
    LICENSE_FREE = 'free'
    LICENSE_PREMIUM = 'premium'
    LICENSE_CHOICES = [
        (LICENSE_FREE, 'Free'),
        (LICENSE_PREMIUM, 'Premium'),
    ]
    
    # Mining Algorithm Choices
    ALGORITHM_ALPHA = 'alpha'
    ALGORITHM_HEURISTICS = 'heuristics'
    ALGORITHM_INDUCTIVE = 'inductive'
    ALGORITHM_CHOICES = [
        (ALGORITHM_ALPHA, 'Alpha Miner'),
        (ALGORITHM_HEURISTICS, 'Heuristics Miner'),
        (ALGORITHM_INDUCTIVE, 'Inductive Miner'),
    ]
    
    roles = models.ManyToManyField(Role, blank=True, related_name='users')
    
    # License fields
    license_type = models.CharField(
        max_length=20,
        choices=LICENSE_CHOICES,
        default=LICENSE_FREE,
        help_text="User's license type"
    )
    
    license_activated_at = models.DateTimeField(
        null=True,
        blank=True,
        help_text="When premium license was activated"
    )
    
    license_expires_at = models.DateTimeField(
        null=True,
        blank=True,
        help_text="When premium license expires (null = lifetime)"
    )
    
    # Limitations for FREE users (can be overridden per user)
    max_log_rows = models.IntegerField(
        default=1000,
        help_text="Maximum number of rows in event log (0 = unlimited)"
    )
    
    allowed_algorithms = models.JSONField(
        default=list,
        blank=True,
        help_text="List of allowed mining algorithms (empty = all allowed)"
    )
    
    max_projects = models.IntegerField(
        default=3,
        help_text="Maximum number of projects (0 = unlimited)"
    )

    @property
    def is_admin(self) -> bool:
        return self.roles.filter(name='Admin').exists()

    @property
    def is_analyst(self) -> bool:
        return self.roles.filter(name='Analyst').exists()
    
    @property
    def is_premium(self) -> bool:
        """Check if user has active premium license"""
        if self.license_type != self.LICENSE_PREMIUM:
            return False
        
        # Check if license is expired
        if self.license_expires_at and self.license_expires_at < timezone.now():
            return False
        
        return True
    
    @property
    def is_free(self) -> bool:
        """Check if user is on free plan"""
        return not self.is_premium
    
    def activate_premium_license(self, duration_days=None):
        """
        Activate premium license for user
        duration_days: None for lifetime, otherwise number of days
        """
        self.license_type = self.LICENSE_PREMIUM
        self.license_activated_at = timezone.now()
        
        if duration_days:
            self.license_expires_at = timezone.now() + timedelta(days=duration_days)
        else:
            self.license_expires_at = None  # Lifetime
        
        # Remove limitations for premium users
        self.max_log_rows = 0  # Unlimited
        self.allowed_algorithms = []  # All allowed
        self.max_projects = 0  # Unlimited
        
        self.save()
    
    def get_allowed_algorithms(self):
        """Get list of allowed algorithms for this user"""
        if self.is_premium or not self.allowed_algorithms:
            # Premium users or users with empty restrictions get all algorithms
            return [choice[0] for choice in self.ALGORITHM_CHOICES]
        return self.allowed_algorithms
    
    def can_use_algorithm(self, algorithm):
        """Check if user can use specific algorithm"""
        return algorithm in self.get_allowed_algorithms()
    
    def __str__(self):
        return f"{self.username} ({self.get_license_type_display()})"


class LicenseCode(models.Model):
    """Premium license activation codes"""
    code = models.CharField(
        max_length=50,
        unique=True,
        help_text="License activation code"
    )
    
    is_active = models.BooleanField(
        default=True,
        help_text="Whether this code can be used"
    )
    
    is_used = models.BooleanField(
        default=False,
        help_text="Whether this code has been used"
    )
    
    used_by = models.ForeignKey(
        User,
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name='used_license_codes',
        help_text="User who used this code"
    )
    
    used_at = models.DateTimeField(
        null=True,
        blank=True,
        help_text="When the code was used"
    )
    
    duration_days = models.IntegerField(
        null=True,
        blank=True,
        help_text="License duration in days (null = lifetime)"
    )
    
    created_at = models.DateTimeField(auto_now_add=True)
    created_by = models.ForeignKey(
        User,
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name='created_license_codes',
        help_text="Admin who created this code"
    )
    
    notes = models.TextField(
        blank=True,
        help_text="Internal notes about this code"
    )
    
    class Meta:
        ordering = ['-created_at']
        verbose_name = 'License Code'
        verbose_name_plural = 'License Codes'
    
    def __str__(self):
        status = "Used" if self.is_used else ("Active" if self.is_active else "Inactive")
        return f"{self.code} ({status})"
    
    def use_code(self, user):
        """Mark code as used and activate license for user"""
        if self.is_used:
            raise ValueError("This code has already been used")
        
        if not self.is_active:
            raise ValueError("This code is not active")
        
        # Activate premium for user
        user.activate_premium_license(self.duration_days)
        
        # Mark code as used
        self.is_used = True
        self.used_by = user
        self.used_at = timezone.now()
        self.save()
        
        return True
